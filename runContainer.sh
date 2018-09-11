  exec docker run \
 	  --name anaco \
          --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix\
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
          -t \
	  anaconda3_reinforcement_learning /bin/bash -c "cd /mnt/shared && python /mnt/shared/chapter01/tic_tac_toe.py"
    exit $