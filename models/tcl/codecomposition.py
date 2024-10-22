import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast

from models.builder import MODELS
from models.tcl.mi import InfoNCE
import us
import torch.distributed as dist

from models.tcl.noun_decomposition import ImageDecomposition
from models.tcl.decoders import TextDecoder, ImgFeatureDncoder
from sclip import tokenize
from models.tcl.encoders import ImgFeatureEncoder
import shared


def handle_padded_tokens(
    masks: torch.Tensor,
    text_indices: torch.Tensor,
    *,
    sos_token_value=None,
    eos_token_value=None,
    padded_token_value=None
):
    """ Handling

    params:
        masks: torch.Tensor of shape (B, L), the original mask
        text_indices: torch.Tensor of shape (B, ), the indices of eos token
        sos_token_value: the value of the start of sentence token to be updated
        eos_token_value: the value of the end of sentence token to be updated
        padded_token_value: the value of the tokens after the eos token to be updated
    returns:
        The updated mask
    """
    if sos_token_value is not None:
        update_mask = torch.zeros_like(masks)
        update_mask[:, 0] = 1.
        masks = masks * (1 - update_mask) + sos_token_value * update_mask

    if eos_token_value is not None:
        update_mask = torch.zeros_like(masks)
        for i, text_index in enumerate(text_indices):
            update_mask[i, text_index] = 1.
        masks = masks * (1 - update_mask) + eos_token_value * update_mask

    if padded_token_value is not None:
        update_mask = torch.zeros_like(masks)
        for i, text_index in enumerate(text_indices):
            update_mask[i, text_index + 1:] = 1.
        masks = masks * (1 - update_mask) + eos_token_value * update_mask

    return masks


def highlight_txt(tokens, txt_mask, bg_txt):
    """
     Highlighting word process

    params:
        tokens: (L, B, C)
        txt_mask: (L, B)
        bg_txt: (L, 1, C)
    """
    fgm = txt_mask[:, :, None]
    output = tokens * fgm + bg_txt * (1 - fgm)
    return output


@MODELS.register_module()
class ImageTextCoDecomposition(ImageDecomposition):
    def __init__(
        self,
        w_hcl,
        w_tseg,
        use_word_highlighting_prompt,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.w_hcl = w_hcl
        self.w_tseg = w_tseg

        self.img_fea_encoder = ImgFeatureEncoder()
        self.img_fea_dncoder = ImgFeatureDncoder()
        self.bce_loss = nn.BCELoss()
        self.L1_loss = nn.L1Loss(reduce='mean')
        self.pooling = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def encode_text_features(self, caption):
        text_token_ids = tokenize(caption, context_length=77, truncate=True)
        text_token_ids = text_token_ids.cuda()

        _, text_hidden_embs = self.frozen_clip.encode_text(text_token_ids, True)

        text_tokens, text_indices = self.frozen_clip.get_word_tokens(
            text_token_ids)
        text_tokens = text_tokens.permute(1, 0, 2)
        return {
            "text_hidden_embs": text_hidden_embs,
            "text_tokens": text_tokens,
            "text_indices": text_indices,
        }

    def scene_category(self, img_feature, caption, noun_emb, all_nouns):
        ret = {}

        # img_feature:[64, 512, 56, 56]   noun_emb:[128, 512]
        text_token_ids = tokenize(caption, context_length=77, truncate=True)
        text_token_ids = text_token_ids.cuda()
        # cap_feature = self.frozen_clip.encode_text(text_token_ids, False) 
        cap_feature, _ = self.frozen_clip.encode_text(text_token_ids, True)        # caption_feature[64, 512]

        img_feature_emb = self.img_fea_encoder(img_feature)
        img_feature_emb_clone = img_feature_emb 
        img_feature_emb = self.pooling(img_feature_emb)          # [64, 512, 1, 1]
        img_feature_emb = img_feature_emb.squeeze(-2).squeeze(-1)   # [64, 512]
        
        ret["img_cap_loss"] = self.L1_loss(cap_feature, img_feature_emb) * 0.5

        # 实例化decoder
        img_feature_emb_clone_decode = self.img_fea_dncoder(img_feature_emb_clone)
        ret['img_img_loss'] = self.L1_loss(img_feature, img_feature_emb_clone_decode)
 
        # img_feature_emb:[64, 512]    noun_emb:[128, 512]  运算得到[64, 128]
        similarity = torch.einsum("ik, jk-> ij", img_feature_emb, noun_emb).to(torch.float32)
        ''' if shared.step != 0 and shared.step % 5000 == 0:
            with open("/mnt/Disk16T/lxl/zjp/Image-Text-Co-Decomposition-main-0809-wzy/matrix_bg/nouns_19.txt", "w", encoding="utf-8") as file:
                file.write("\n".join(all_nouns))
            similarity_save = np.array(similarity.detach().cpu())
            np.save('/mnt/Disk16T/lxl/zjp/Image-Text-Co-Decomposition-main-0809-wzy/matrix_bg/array_19.npy', similarity_save)
        '''
        similarity = torch.sigmoid(similarity)
        mapping = {}
        half_len = len(all_nouns) // 2
        for index, word in enumerate(all_nouns[:half_len]):
            mapping.setdefault(word, []).append(index)
            mapping.setdefault(all_nouns[index + half_len], []).append(index)

        matrix = torch.zeros((similarity.shape[0], similarity.shape[1]), dtype=torch.float)
        weight_bceloss = torch.ones((similarity.shape[0], similarity.shape[1]), dtype=torch.float).to(similarity.device)

        for index, word in enumerate(all_nouns):
            matrix[mapping[word], index] = 1
            weight_bceloss[mapping[word], index] = 5
        truth_label = matrix.to(similarity.device)
        
        bce_loss_new = nn.BCELoss(weight=weight_bceloss)
        with torch.cuda.amp.autocast(enabled=False):  
            ret["incidence_matrix_loss"] = bce_loss_new(similarity, truth_label) * 2 
        mapping.clear()
        return similarity, ret
    

    def forward(self, image, category, mask_gt, caption, use_pamr=False):
        num_nouns = len(category)
        all_nouns = sum((noun_list for noun_list in category), [])

        ret = {}  # losses + logs

        decoded_feat_map = self.decode_feature_map(image)
        img_feature = decoded_feat_map
        decoded_feat_map = torch.cat([decoded_feat_map] * num_nouns, dim=0)
        image = torch.cat([image] * num_nouns, dim=0)

        # Build noun embeddings
        noun_embs = self.clip_text_encoder(all_nouns)
        ret["kg_loss"] = self.w_kg * self.cal_kg_loss(noun_embs, all_nouns)

        incidence_matrix, new_ret = self.scene_category(img_feature, caption, noun_embs, all_nouns)
        ret.update(new_ret)

        masks = self.masker(decoded_feat_map, noun_embs, incidence_matrix)

        mask_pos = torch.cat([masks['soft_all'][i, i:i+1, :, :] for i in range(masks['soft_all'].shape[0])], dim=0)
        mask_resize_pos = F.interpolate(mask_pos.unsqueeze(0), size=mask_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        mask_gt = mask_gt.permute(1, 0, 2, 3).reshape(mask_pos.shape[0], mask_gt.shape[-2], mask_gt.shape[-1])
        # with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        
        with autocast(enabled=False):
            ret['bce_loss'] = self.bce_loss(mask_resize_pos.float(), mask_gt.float()) * 2
        # 2   3            1.5
        mask_resize_gt = F.interpolate(mask_gt.unsqueeze(0), size=mask_pos.shape[-2:], mode='nearest').squeeze(0)
        # with autocast(enabled=False):
        #     ret['resize_becloss'] = self.bce_loss(mask_pos.float(), mask_resize_gt.float()) * 2

        mask_pos_1 = mask_pos[:64, :, :].view(64, -1)
        mask_pos_2 = mask_pos[64:, :, :].view(64, -1)
        mask_gt_1 = mask_resize_gt[:64, :, :].view(64, -1)
        mask_gt_2 = mask_resize_gt[64:, :, :].view(64, -1)
        mask_pos_new = torch.bmm(mask_pos_1.unsqueeze(2), mask_pos_2.unsqueeze(1))
        mask_gt_new = torch.bmm(mask_gt_1.unsqueeze(2).float(), mask_gt_2.unsqueeze(1).float())
        with autocast(enabled=False):
            ret['new_loss'] = self.bce_loss(mask_pos_new.float(), mask_gt_new.float()) * 2.5
            # 2.5    7      2
        
        new_ret, fg_image_emb = self.cal_iseg_loss(
            image,
            masks,
            decoded_feat_map,
            noun_embs,
        )

        ret.update(new_ret)
        

        if use_pamr:
            masks["soft_pos"] = self.apply_pamr(image, masks["soft_pos"])

        records = {
            "image": image,
            "masks": masks,
        }

        return ret, records