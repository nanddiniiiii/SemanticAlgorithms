# Kaggle datasets

Attach these datasets to each notebook before running the common image experiments:

| Dataset | Kaggle input | Role |
|---|---|---|
| DIV2K HR | `soumikrakshit/div2k-high-resolution-images` | 800 training images and 100 development images |
| Flickr2K | `daehoyang/flickr2k` | 2,650 training entries |
| Kodak24 | `drxinchengzhu/kodak24` | 24 final test images |

The verified common manifest retains 59 exact duplicate training entries; no
exact duplicate was found involving development or test. See
`Docs/TEAM_SUITE_VALIDATION.md` for the scope and manifest identity.

The shared semantic-aware codebook also needs an accessible Kaggle dataset
containing its frozen `classifier.pt`. SQ-GAN needs author component weights
and a frozen segmentation model; its custom-data path uses pseudo-labels.
Requirements and access can differ between Kaggle accounts. Do not place
credentials or private dataset contents in this repository.
