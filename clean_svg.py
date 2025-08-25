import xml.etree.ElementTree as ET

# 注册SVG命名空间
ET.register_namespace("", "http://www.w3.org/2000/svg")

def clean_svg(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 命名空间
    inkscape_ns = "http://www.inkscape.org/namespaces/inkscape"
    sodipodi_ns = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
    xml_ns = "http://www.w3.org/XML/1998/namespace"

    namespaces_to_strip = [inkscape_ns, sodipodi_ns, xml_ns]

    def strip_namespaces(elem):
        """去掉不需要的命名空间属性"""
        attribs_to_remove = [
            k for k in elem.attrib
            if any(ns in k for ns in namespaces_to_strip)
        ]
        for k in attribs_to_remove:
            del elem.attrib[k]

    # 遍历所有元素
    for elem in root.iter():
        strip_namespaces(elem)

        # 如果是 group (g) 或 layer，统一转成 group
        if elem.tag.endswith("g"):
            # 如果有 inkscape:label，就用它替换 id
            label_key = f"{{{inkscape_ns}}}label"
            groupmode_key = f"{{{inkscape_ns}}}groupmode"

            label_val = elem.attrib.get(label_key)
            if label_val:
                elem.set("id", label_val)   # 设置 id = label

            # 删除 inkscape 的特殊属性
            elem.attrib.pop(label_key, None)
            elem.attrib.pop(groupmode_key, None)

    # 移除 <metadata> 和 <defs>
    for bad_tag in ["metadata", "defs"]:
        for bad in root.findall(f".//{{http://www.w3.org/2000/svg}}{bad_tag}"):
            parent = root
            for p in tree.iter():
                if bad in list(p):
                    parent = p
                    break
            parent.remove(bad)

    # 移除空 group
    def remove_empty_groups(elem):
        for g in list(elem):
            remove_empty_groups(g)
            if g.tag.endswith("g") and len(g) == 0 and not g.attrib:
                elem.remove(g)

    remove_empty_groups(root)

    # 写回文件
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    clean_svg("input.svg", "cleaned.svg")
    print("SVG 已清理完成，输出到 cleaned.svg")
