# TODO – Future Enhancements and Investigations

## Alignment Strategies
- Implement intra-subject temporal alignment prior to cross-subject standardization.
- Explore velocity-based alignment without relying solely on binary masks.

## Segmentation Robustness
- Introduce segmentation correction steps for frames where alignment results are poor.
- Refine the radius estimation function for cases with extremely small (<5 px) or large (>13 px) masks.

## Zoom-Based Labeling Experiments
- Investigate label segmentation on linearly zoomed and cropped patches.
- After segmentation, restore labels to original frame size using inverse transforms.
- Compare:
  - Training models directly on zoomed images.
  - Labeling original frames and zooming post hoc.
- Optionally train two separate networks and compare outcomes to inform best practice.

## Model and Pipeline Improvements
- Use the base DeepFlow segmentation model as a starting point.
- Explore few-shot or sample-correction learning techniques to improve segmentation quality.
- Improve DeepFlow’s mask encoding/decoding process.
- Evaluate the effect of training the network at native resolution or higher (e.g., 192×192).

## Noise Reduction and Outlier Handling
- Discard 2–3 outer rings from circular masks to reduce peripheral segmentation noise.
- Flag subject file `20213` as an outlier and treat accordingly in template alignment or dataset preparation.
