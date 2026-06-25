#!/bin/bash
# Make a YouTube-ready MP4 from a still image + an audio track.
#   bash make_video.sh <image.png> <audio.mp3> <out.mp4>
IMG="$1"; AUD="$2"; OUT="$3"
ffmpeg -y -loglevel error -loop 1 -i "$IMG" -i "$AUD" \
  -c:v libx264 -tune stillimage -pix_fmt yuv420p -vf "scale=1920:1080" \
  -c:a aac -b:a 192k -shortest "$OUT"
echo "VIDEO: $OUT"
