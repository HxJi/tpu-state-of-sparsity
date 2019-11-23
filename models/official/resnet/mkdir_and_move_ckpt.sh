for i in {1..102}
do
  j=$[i*1251]
  mkdir model.ckpt-$j
  mv model.ckpt-$j.* model.ckpt-$j
done 