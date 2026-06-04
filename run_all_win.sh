#!/usr/bin/env bash
# Self-healing Windows/conda reproduction runner — mirrors submitv2.sh order.
# OOM fallback: isvl/CPR training+inference retry at batch 8 -> 4 -> 2.
# On unrecoverable failure: prints PIPELINE_FAILED and exits 1 (so we get notified).
cd "e:/Dev/MVTECAD2_PostProc"
export CUDA_VISIBLE_DEVICES=0            # GPU0 = clean 3080 (GPU1 has the display)
export PYTHONUTF8=1                      # repo scripts print Chinese; avoid cp949 UnicodeEncodeError on redirect
export PYTHONIOENCODING=utf-8
PY="D:/anaconda/envs/cvprw/python.exe"

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
step() { echo ""; echo "=================================================================="; echo "### $* :: $(ts)"; echo "=================================================================="; }
die()  { echo ""; echo "XXXXXX PIPELINE_FAILED at: $* :: $(ts) XXXXXX"; exit 1; }

isvl_train() {  # $1=category
  local cat="$1"
  for bs in 8 4 2; do
    step "[TRAIN] $cat (batch $bs)"
    if "$PY" isvl.py --item_list "$cat" --total_epochs 10 --batch_size "$bs"; then
      echo "### [TRAIN] $cat OK @batch $bs :: $(ts)"; return 0
    fi
    echo "!!! [TRAIN] $cat failed @batch $bs (likely OOM) — retry smaller :: $(ts)"
  done
  die "[TRAIN] $cat (all batch sizes failed)"
}

isvl_val() {  # $1=category
  local cat="$1"
  for bs in 8 4 2; do
    step "[VAL] $cat (batch $bs)"
    if "$PY" isvl.py --item_list "$cat" --total_epochs 10 --phase val --batch_size "$bs"; then
      echo "### [VAL] $cat OK @batch $bs :: $(ts)"; return 0
    fi
    echo "!!! [VAL] $cat failed @batch $bs — retry smaller :: $(ts)"
  done
  die "[VAL] $cat (all batch sizes failed)"
}

cpr_train() {  # $1=category $2=steps
  local cat="$1"; local steps="$2"
  for bs in 8 4 2; do
    step "[CPR train] $cat steps=$steps (batch $bs)"
    if "$PY" train.py -fd log/foreground/foreground_mvtec_test_vial_fruit --steps "$steps" -tps "$steps" \
         --data-dir log/synthesized/synthesized_mvtec_test_vial_fruit \
         --retrieval-dir log/retrieval/retrieval_mvtec_test_vial_fruit \
         --dataset-name mvtec_test_vial_fruit --category "$cat" --batch-size "$bs"; then
      echo "### [CPR train] $cat OK @batch $bs :: $(ts)"; return 0
    fi
    echo "!!! [CPR train] $cat failed @batch $bs — retry smaller :: $(ts)"
  done
  die "[CPR train] $cat (all batch sizes failed)"
}

step "PIPELINE START :: $(ts)"

step "[1] Tiling (image_splitter)"
"$PY" 1_image_splitter.py            || die "1_image_splitter"
step "[1] Restructure (vial/fruit_jelly for CPR)"
"$PY" 1_restructure_mvtec_dataset.py || die "1_restructure"

# ---- INP-Former training (6 categories) ----
for c in can fabric rice sheet_metal wallplugs walnuts; do isvl_train "$c"; done

# ---- CPR branch (fruit_jelly, vial) ----
step "[CPR] generate foreground"
"$PY" tools/generate_foreground.py -lp log/foreground/foreground_mvtec_test_vial_fruit \
  --dataset-name mvtec_test_vial_fruit --layer features.denseblock1 -pm DenseNet || die "generate_foreground"
step "[CPR] generate retrieval"
"$PY" tools/generate_retrieval.py -lp log/retrieval/retrieval_mvtec_test_vial_fruit \
  --dataset-name mvtec_test_vial_fruit --layer features.denseblock1 -pm DenseNet || die "generate_retrieval"
step "[CPR] synthesize fruit_jelly"
"$PY" tools/generate_synthesize_hand.py --output_dir log/synthesized/synthesized_mvtec_test_vial_fruit \
  --dataset-name mvtec_test_vial_fruit --normal_dir data/mvtec_test_vial_fruit \
  --mask_dir log/foreground/foreground_mvtec_test_vial_fruit \
  --num_per_image 5 --resize 640 --seed 42 --category fruit_jelly || die "synthesize fruit_jelly"
step "[CPR] synthesize vial"
"$PY" tools/generate_synthesize_hand.py --output_dir log/synthesized/synthesized_mvtec_test_vial_fruit \
  --dataset-name mvtec_test_vial_fruit --normal_dir data/mvtec_test_vial_fruit \
  --mask_dir log/foreground/foreground_mvtec_test_vial_fruit \
  --num_per_image 3 --resize 640 --seed 66 --category vial || die "synthesize vial"

cpr_train fruit_jelly 2000
cpr_train vial 1300

# ---- Inference ----
for c in can fabric rice sheet_metal wallplugs walnuts; do isvl_val "$c"; done

step "[CPR] test fruit_jelly"
"$PY" test_new.py -fd log/foreground/foreground_mvtec_test_vial_fruit \
  --checkpoints log/chekpoints/chekpoints_mvtec_test_vial_fruit_True/fruit_jelly/02000.pth \
  -rd log/retrieval/retrieval_mvtec_test_vial_fruit -dn mvtec_test_vial_fruit --sub-categories fruit_jelly || die "test fruit_jelly"
step "[CPR] test vial"
"$PY" test_new.py -fd log/foreground/foreground_mvtec_test_vial_fruit \
  --checkpoints log/chekpoints/chekpoints_mvtec_test_vial_fruit_True/vial/01300.pth \
  -rd log/retrieval/retrieval_mvtec_test_vial_fruit -dn mvtec_test_vial_fruit --sub-categories vial || die "test vial"

# ---- Post-processing (steps 2-8) ----
step "[2] reconstruction (stitch patches)"; "$PY" 2_image_reconstruction.py        || die "2_reconstruction"
step "[3] replace_and_rename";              "$PY" 3_replace_and_rename_folders.py    || die "3_rename"
step "[4] threshold_mapv2";                 "$PY" 4_threshold_mapv2.py               || die "4_threshold"
step "[5] postproc fabric";                 "$PY" 5_post_image_process.py            || die "5_fabric"
step "[5] erode fruit_jelly";               "$PY" 5_erode_image.py                   || die "5_erode"
step "[5] postproc wallnuts";               "$PY" 5_post_image_process_wallnuts.py   || die "5_wallnuts"
step "[6] replace_and_rename";              "$PY" 6_replace_and_rename_folders.py     || die "6_rename"
step "[7] convert_tiff_to_float16";         "$PY" 7_convert_tiff_to_float16.py        || die "7_float16"
step "[8] check_and_prepare_upload";        "$PY" 8_check_and_prepare_data_for_upload.py "./results/" || die "8_package"

step "ALL DONE :: $(ts)"
echo "PIPELINE_SUCCESS"
ls -lh results.tar.gz 2>/dev/null || echo "(results.tar.gz not found — inspect step 8 output)"
