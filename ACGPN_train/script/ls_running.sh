# this command get name of running experiments and help for kill
# MAGIC ~~~
ps ux | grep name | grep -v grep | sed 's/^.*name *\([^ ]*\) .*/\1/' | uniq | grep '^[a-zA-Z]'