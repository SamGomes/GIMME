WID=`xdotool search --class gwenview`
xdotool windowactivate $WID

watch -n 2 'Rscript plotGeneratorEvl.r; xdotool key F5'
