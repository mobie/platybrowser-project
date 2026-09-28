# README

`2025_scripts/adding_HCR/staging_from_kreshuk.sh` will symlink from Alyona's test install at `/g/kreshuk/buglakova/data/platy_registration/platybrowser-smfish-project/data/1.0.1/images/bdv-n5` t osee what is going on.


I am going back to project root and running

    ./2025_scripts/upload_Alyona_local_n5_to_s3.py \
      -i /scratch/cros/platybrowser_staging/HCR_combined \
      -o /home/cros/bioinformatics/platybrowser-project-2025/data/platybrowser_6dpf/images/bdv-n5-s3/HCR_combined \
      -b platybrowser-2025 \
      -e https://s3.embl.de \
      -r us-west-2 \
      -p HCR_combined \
      --dry-run

Individual samples, not for the paper because too big, remove later.

    ./2025_scripts/upload_Alyona_local_n5_to_s3.py \
      -i /scratch/cros/platybrowser_staging/HCR_individual \
      -o /home/cros/bioinformatics/platybrowser-project-2025/data/platybrowser_6dpf/images/bdv-n5-s3/HCR_individual \
      -b platybrowser-all-hcrs-temp \
      -e https://s3.embl.de \
      -r us-west-2 \
      -p HCR_individual \
      --dry-run