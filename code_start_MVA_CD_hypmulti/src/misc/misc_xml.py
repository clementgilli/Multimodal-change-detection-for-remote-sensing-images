# General
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring
from xml.dom import minidom
from xml.dom.minidom import parse, parseString
import ast


# Imaging

# Local


def dict_to_xml(tag: str, d: dict, indent=4):
    """
    Code given by Xonxt on GitHub:
    https://gist.github.com/phalt/b5cdf3d3d5e91879fd09429adb070244
    """
    elem = Element(tag)

    def _d_to_x(elem, d):
        for key, val in d.items():

            if isinstance(val, dict):
                elem.append(_d_to_x(Element(key), val))

            elif isinstance(val, list):
                for l in val:
                    elem.append(_d_to_x(Element(key), l))
            else:
                child = Element(key)
                child.text = str(val)
                elem.append(child)

        return elem

    # prettify
    xml_str = tostring(_d_to_x(elem, d)).decode("utf-8")
    xml_str = minidom.parseString(xml_str)
    xml_str = xml_str.toprettyxml(indent=' ' * indent)

    return xml_str


def save_xml_str(path: str, xml_str: str):
    f = open(path, "w")
    f.write(xml_str)
    f.close()



def xml_dict(node, path: str):
    """
    Source - https://stackoverflow.com/a/3217550
    Posted by jsbueno, modified by community. See post 'Timeline' for change history
    Retrieved 2026-02-12, License - CC BY-SA 3.0
    """
    d = {}

    name_prefix = path + ("." if path else "") + node.tag
    numbers = set()
    for similar_name in d.keys():
        if similar_name.startswith(name_prefix):
            numbers.add(int (similar_name[len(name_prefix):].split(".")[0] ) )
    if not numbers:
        numbers.add(0)
    index = max(numbers) + 1
    name = name_prefix + str(index)
    d[name] = node.text + "<...>".join(childnode.tail
                                         if childnode.tail is not None else
                                         "" for childnode in node)
    for childnode in node:
        xml_dict(childnode, name, d)
    return d


def load_xml_to_dict(path: str):

    xml_data = open(path, "r").read()
    root = ET.XML(xml_data)  # Parse XML
    return root_xml_to_dict(root)


def root_xml_to_dict(root: ET.Element) -> dict:
    d = {}

    for i, child in enumerate(root):
        subchilds = [subchild for subchild in child]

        if len(subchilds) == 0:
            try:
                d[child.tag] = ast.literal_eval(child.text)  # string to any type
            except ValueError:
                d[child.tag] = child.text
            except SyntaxError:
                d[child.tag] = child.text

        else:
            d[child.tag] = root_xml_to_dict(child)

    return d
