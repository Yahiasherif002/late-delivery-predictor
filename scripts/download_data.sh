#!/usr/bin/env sh
# Downloads the Olist dataset (CC BY-NC-SA 4.0) with the Kaggle CLI into data/raw.
set -e
mkdir -p data/raw
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
