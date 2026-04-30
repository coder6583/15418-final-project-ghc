with open("mic_data.txt") as f:
    lines = f.readlines()

# skip first 2 lines
lines = lines[2:]
print(lines)

rows = []
for i in range(4000):
    nums = list(map(int, lines[i].split()))
    nums = (nums + [0, 0, 0, 0])[:4]
    rows.append(nums)

with open("data.h", "w") as f:
    f.write("static const int32_t data[512][4] = {\n")
    for r in rows:
        f.write(f"    {{ {r[0]}, {r[1]}, {r[2]}, {r[3]} }},\n")
    f.write("};\n")
