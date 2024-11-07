# ------------------------------------------------------------------------------
# CoDe
# Copyright (C) 2024 by Ji-Jia Wu. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from TCL (https://github.com/kakaobrain/tcl)
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
import torch.nn.functional as F
from mmseg.models import EncoderDecoder
from utils import get_logger
from mmseg.ops import resize
import numpy as np
from PIL import Image
import os


class TCLSegInference(EncoderDecoder):
    def __init__(
        self,
        model,
        text_embedding,
        kp_branch_text_embedding,
        with_bg,
        test_cfg=dict(),
        pamr=False,
        bg_thresh=0.5,
        kp_w=0.3,
        **kwargs,
    ):
        super(EncoderDecoder, self).__init__()  # init BaseSegmenter (parent of EncoderDecoder)

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.kp_w = kp_w

        self.model = model
        self.register_buffer("text_embedding", text_embedding)
        self.register_buffer("kp_branch_text_embedding", kp_branch_text_embedding)
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)

        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building TCLSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}, kp_w={kp_w}"
        )

    def generate(self, masks, img_metas):
        num_classes = self.num_classes 
        preds_mask = masks.squeeze(0).cpu().numpy()
        preds_mask = F.softmax(torch.from_numpy(preds_mask), dim=0).argmax(0).cpu().numpy()

        palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170,30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180],
            [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        seg_img = np.zeros((preds_mask.shape[0], preds_mask.shape[1], 3))
        for c in range(num_classes):
            seg_img[:, :, 0] += ((preds_mask[:, :] == c) * palette[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((preds_mask[:, :] == c) * palette[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((preds_mask[:, :] == c) * palette[c][2]).astype('uint8')
        colorized_mask = Image.fromarray(np.uint8(seg_img))
        # image_file = img_metas[0]['ori_filename'].split('.')[0]
        # city专用
        image_file = img_metas[0]['ori_filename'].split('.')[0].split('/')[-1]
        colorized_mask.save(os.path.join('/mnt/Disk16T/lxl/zjp/inference_show/exp_41/city/masks/' + image_file + '.png'))

        
        img_1 = Image.open(img_metas[0]['filename'])
        img_2 = Image.open("/mnt/Disk16T/lxl/zjp/val_gt/city/" + image_file + ".png")
        images = [img_1, img_2, colorized_mask]
        total_width = sum(image.width for image in images) + (len(images) - 1) * 30
        max_height = max(image.height for image in images) + 30

        new_image = Image.new('RGB', (total_width, max_height), color='white')

        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.width + 30

        new_image.save(os.path.join('/mnt/Disk16T/lxl/zjp/inference_show/exp_41/city/combined/' + image_file + '.png'))


    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        assert img.shape[0] == 1, "batch size must be 1"

        # masks [B, N, H, W]
        # simmap [B, N, H//4, W//4]
        # soft mask (logit-like) is required
        masks, simmap = self.model.generate_masks(
            img,
            self.text_embedding,
            self.kp_branch_text_embedding,
            apply_pamr=self.pamr,
            kp_w=self.kp_w,
        )

        B, N, H, W = masks.shape

        if self.with_bg:
            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
        return masks
    

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        self.generate(preds, img_meta)
        return preds
