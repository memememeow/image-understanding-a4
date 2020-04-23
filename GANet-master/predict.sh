
python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='./car_imgs/' \
                  --test_list='./lists/q2_car.list' \
                  --save_path='./' \
                  --kitti2015=1 \
                  --resume='./kitti2015_final.pth'
exit

python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='./car_imgs/' \
                  --test_list='lists/q2_car.list' \
                  --save_path='./' \
                  --kitti=1 \
                  --resume='./kitti2012_final.pth'
