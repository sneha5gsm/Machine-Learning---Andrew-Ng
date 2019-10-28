# function [h, display_array] = displayData(X, example_width)
#DISPLAYDATA Display 2D data in a nice grid
#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid. It returns the figure handle h and the
#   displayed array if requested.
#
# Set example_width automatically if not passed in
# if ~exist('example_width', 'var') || isempty(example_width)
# 	example_width = round(sqrt(size(X, 2)));
# end
#
# Gray Image
# colormap(gray);
#
# Compute rows, cols
# [m n] = size(X);
# example_height = (n / example_width);
#
# Compute number of items to display
# display_rows = floor(sqrt(m));
# display_cols = ceil(m / display_rows);
#
# Between images padding
# pad = 1;
#
# Setup blank display
# display_array = - ones(pad + display_rows * (example_height + pad), ...
#                        pad + display_cols * (example_width + pad));
#
# Copy each example into a patch on the display array
# curr_ex = 1;
# for j = 1:display_rows
# 	for i = 1:display_cols
# 		if curr_ex > m,
# 			break;
# 		end
# 	 Copy the patch
#
# 	 Get the max value of the patch
# 		max_val = max(abs(X(curr_ex, :)));
# 		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
# 		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
# 						reshape(X(curr_ex, :), example_height, example_width) / max_val;
# 		curr_ex = curr_ex + 1;
# 	end
# 	if curr_ex > m,
# 		break;
# 	end
# end
#
# Display Image
# h = imagesc(display_array, [-1 1]);
#
# Do not show axis
# axis image off
#
# drawnow;
#
# end
import numpy as np
import math
import matplotlib.pyplot as plt
def displayData(x_modified, example_width):
	# x_modified=[]
	[m, n] = np.shape(x_modified)
	if example_width==None:
		example_width = round(math.sqrt(n))
	example_height = int(n / example_width)
	display_rows = math.floor(math.sqrt(m))
	display_cols = math.ceil(m / display_rows)
	pad = 1
	# x_modified = x
	# for i in range(m):
	# 	temp = np.reshape(x[i], (example_height, example_width))
	# 	temp = list(np.transpose(temp).ravel())
	# 	x_modified.append(temp)
		# plt.imshow(np.reshape(temp, (example_height, example_width)))
		# plt.show()

	# print(example_width)
	# print(example_height)
	# print(display_cols)
	# print(display_rows)
	display_array = [[1] * (display_cols)* (example_width) for _ in range((display_rows) * (example_height))]
	curr_ex = 0
	for j in range(display_rows):
		parameter = display_cols
		if (curr_ex + display_cols) >= m:
			parameter = m - curr_ex
		for i in range(parameter):
			for k in range(example_height):
				# print(np.shape(display_array[(j*example_height) + k]))
				# print(np.shape(display_array[(j*example_height) + k][:i*example_width]))
				# print(np.shape(x_modified[(j*display_cols)+i][k*example_width: k*example_width+example_width]))
				# print(np.shape(display_array[(j * example_height) + k][i * example_width + example_width:]))
				#
				# print(display_array[(j * example_height) + k])
				# print(np.shape(display_array[(j * example_height) + k][:i * example_width]))
				# print(x_modified[(j * display_cols) + i][k * example_width: k * example_width + example_width])
				# print(display_array[(j * example_height) + k][i * example_width + example_width:])
				display_array[(j*example_height) + k] = display_array[(j*example_height) + k][:i*example_width] + list(x_modified[(j*display_cols)+i][k*example_width: k*example_width+example_width]) + display_array[(j*example_height) + k][i*example_width+example_width:]
				# plt.imshow(display_array)
				# plt.title('Compressed, with #d colors.')
				# plt.show()
				# print('--------------------------------------')
		curr_ex = curr_ex+display_cols
	# print(display_array)
	# print(len(display_array))
	# print(len(display_array[0]))
	# print(display_array[0])
	plt.imshow(display_array)
	plt.title('Compressed, with #d colors.')
	plt.show()
	# for j in range(display_rows):
	# 	for i in range(display_cols):
	# 		if curr_ex > m:
	# 			break
	# 		max_val = max(abs(X(curr_ex, :)));
	# 		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
	# 					  pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
	# 						reshape(X(curr_ex, :), example_height, example_width) / max_val;
	# 		curr_ex = curr_ex + 1
	# 	if curr_ex > m:
	# 		break

