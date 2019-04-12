#!/usr/bin/env bash
rsync -r cast setup.py $1:repo/cast

ssh $1 << 'EOF'
	case $(hostname -s) in
		login0*)
			module add devel/python/3.5.1
			;;
		*)
			;;
	esac

	cd repo/cast
	python3 setup.py install --user 
EOF

