from visualdl import LogReader


def get_tags(reader, includes=["ACC", "RMSE"]):
    tags = reader.get_tags()["scalar"]

    result_tags = []
    for tag in tags:
        for include in includes:
            if include in tag:
                result_tags.append(tag)
                break
    return result_tags


def get_tag_data(reader, tag):
    "Get the tag data of a tag"
    data = reader.get_data("scalar", tag)
    return data


file_path1 = "../output/hrrr/vdl_no_weight/vdlrecords.1690973764.log"
file_path2 = "../output/hrrr/vdl_no_weight/vdlrecords.1690973764.log"

reader1 = LogReader(file_path=file_path1)
reader2 = LogReader(file_path=file_path2)
tags = get_tags(reader1)

total_target = 0
greater = 0
less = 0
acc_diff = 0
rmse_diff = 0
for tag in tags:
    tag_data1 = get_tag_data(reader1, tag)
    tag_data2 = get_tag_data(reader2, tag)

    tag_data1 = sorted(tag_data1, key=lambda k: k.id)
    tag_data2 = sorted(tag_data2, key=lambda k: k.id)

    for tag_data1_item, tag_data2_item in zip(tag_data1, tag_data2):
        total_target += 1
        if "RMSE" in tag:
            rmse_diff += tag_data2_item.value - tag_data1_item.value
            if tag_data1_item.value >= tag_data2_item.value:
                greater += 1
            else:
                less += 1
        else:
            acc_diff += tag_data2_item.value - tag_data1_item.value
            if tag_data1_item.value <= tag_data2_item.value:
                greater += 1
            else:
                less += 1
print("Advantages of File 2 over File 1:")
print("Total target:", total_target)
print("Greater target:", greater)
print("Less target:", less)

print("Greater target ratio:", greater / total_target)
print("Less target ratio:", less / total_target)

print("ACC diff:", acc_diff)
print("RMSE diff:", rmse_diff)

print("ACC avg diff:", acc_diff / total_target)
print("RMSE avg diff:", rmse_diff / total_target)
