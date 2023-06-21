The following files are added from
https://github.com/PDillis/stylegan3-fun
into the stylegan3 codebase.

1 ./stylegan3/network_features.py
2 ./stylegan3/torch_utils/gen_utils.py

python projector.py --target '../data/retina_GA/AREDS_2014_images_512x512/test/60746_RE/60746 03 F2 RE RS.jpg' --project-in-wplus --save-video --num-steps=5000 --network ../stylegan3/training-runs/00001-stylegan3-r--gpus1-batch16-gamma5/network-snapshot-002000.pkl