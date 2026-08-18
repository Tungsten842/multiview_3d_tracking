ffmpeg -y -i out4.mp4 -vf "scale=1280:720" -c:v libx264 -crf 6 out4d.mp4

ffmpeg -y -i out13.mp4 -vf "crop=iw/1.2:ih/1.2,scale=1280:720" -c:v libx264 -crf 6 out13d.mp4
