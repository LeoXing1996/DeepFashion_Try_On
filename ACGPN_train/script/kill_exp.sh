name=$1
# echo $name
ps ux | grep $name | awk '{print $2}' | xargs kill