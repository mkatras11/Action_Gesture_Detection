# LFB

[Long-term feature banks for detailed video understanding](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Long-Term_Feature_Banks_for_Detailed_Video_Understanding_CVPR_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

To understand the world, we humans constantly need to relate the present to the past, and put events in context. In this paper, we enable existing video models to do the same. We propose a long-term feature bank---supportive information extracted over the entire span of a video---to augment state-of-the-art video models that otherwise would only view short clips of 2-5 seconds. Our experiments demonstrate that augmenting 3D convolutional networks with a long-term feature bank yields state-of-the-art results on three challenging video datasets: AVA, EPIC-Kitchens, and Charades.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016220-21d90fb3-fd9f-499c-820f-f6c421bda7aa.png" width="800"/>
</div>

## Results and Models

### AVA 2.1

|                                                                          Model                                                                          | Modality |  Pretrained  |                                               Backbone                                               | Input | gpus |   Resolution   |  mAP  |                                                                     log                                                                      |                                                                        json                                                                        |                                                                                                    ckpt                                                                                                     |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :----------: | :--------------------------------------------------------------------------------------------------: | :---: | :--: | :------------: | :---: | :------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py)  |   RGB    | Kinetics-400 | [slowonly_r50_4x16x1](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) | 4x16  |  8   | short-side 256 | 24.11 | [log](https://download.openmmlab.com/mmaction/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210224_125052.log)  | [json](https://download.openmmlab.com/mmaction/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210224_125052.log.json)  |  [ckpt](https://download.openmmlab.com/mmaction/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb_20210224-2ae136d9.pth)  |
| [lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | [slowonly_r50_4x16x1](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) | 4x16  |  8   | short-side 256 | 20.17 | [log](https://download.openmmlab.com/mmaction/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log) | [json](https://download.openmmlab.com/mmaction/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb_20210301-19c330b7.pth) |
| [lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | [slowonly_r50_4x16x1](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) | 4x16  |  8   | short-side 256 | 22.15 | [log](https://download.openmmlab.com/mmaction/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log) | [json](https://download.openmmlab.com/mmaction/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb_20210301-37efcd15.pth) |

## Citation

<!-- [DATASET] -->

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}
```

```BibTeX
@inproceedings{wu2019long,
  title={Long-term feature banks for detailed video understanding},
  author={Wu, Chao-Yuan and Feichtenhofer, Christoph and Fan, Haoqi and He, Kaiming and Krahenbuhl, Philipp and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={284--293},
  year={2019}
}
```
