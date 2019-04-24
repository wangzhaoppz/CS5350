f = open("data_result/HW3_result.txt")

lines = []
for line in f:
	lines.append(line)

f = open("data_result/HW3_result_laTex.txt", "w")
for line in lines:
	f.write(line + "\\newline ")
