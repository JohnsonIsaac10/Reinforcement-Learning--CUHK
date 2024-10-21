win_width = 30
win_height = 20
temp = 0
nx = 0
ny = 0
for nx in range(1, int(win_width / 3) + 1):
    for ny in range(1, int(win_height / 2) + 1):
        number_of_blocks_x = (win_width - 3 * nx + 1)
        number_of_blocks_y = (win_height - 2 * ny + 1)
        temp = number_of_blocks_x * number_of_blocks_y + temp

for nx in range(1, int(win_width / 2) + 1):
    for ny in range(1, int(win_height) + 1):
        number_of_blocks_x = win_width - 2 * nx + 1
        number_of_blocks_y = win_height - ny + 1
        temp = number_of_blocks_y * number_of_blocks_x + temp

for nx in range(1, int(win_width / 3) + 1):
    for ny in range(1, int(win_height) + 1):
        number_of_blocks_x = win_width - 3 * nx + 1
        number_of_blocks_y = win_height - ny + 1
        temp = number_of_blocks_x * number_of_blocks_y + temp

print(temp)

