import os
from ase.db import connect
from ase.io.extxyz import read_xyz
from tqdm import tqdm
import tempfile


def xyz_to_extxyz(
    xyz_path,
    extxyz_path,
    atomic_properties="Properties=species:S:1:pos:R:3",
    molecular_properties=[],
):
    """
    Convert a xyz-file to extxyz.

    Args:
        xyz_path (str): path to the xyz file
        extxyz_path (str): path to extxyz-file
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # ensure valid property string
    atomic_properties = parse_property_string(atomic_properties)
    new_file = open(extxyz_path, "w")
    with open(xyz_path, "r") as xyz_file:
        while True:
            first_line = xyz_file.readline()
            if first_line == "":
                break
            n_atoms = int(first_line.strip("\n"))
            if "Lattice" not in molecular_properties:
                molecular_data = xyz_file.readline().strip("/n").split()
                assert len(molecular_data) == len(molecular_properties), (
                    "The number of datapoints and " "properties do not match!"
                )
                comment = " ".join(
                    [
                        "{}={}".format(prop, val)
                        for prop, val in zip(molecular_properties, molecular_data)
                    ]
                )
            else:
                #只针对分子信息只有能量和晶胞尺寸的情况
                molecular_data = xyz_file.readline().strip("/n").split()
                comment = " ".join(
                    [
                        "{}={}".format(molecular_properties[0], molecular_data[0]),
                        "{}=\"{}\"".format(molecular_properties[1], " ".join(molecular_data[1:]))
                    ]
                )

            new_file.writelines(str(n_atoms) + "\n")
            new_file.writelines(" ".join([atomic_properties, comment]) + "\n")
            for i in range(n_atoms):
                line = xyz_file.readline()
                new_file.writelines(line)
    new_file.close()


def extxyz_to_db(extxyz_path, db_path):
    r"""
    Convertes en extxyz-file to an ase database

    Args:
        extxyz_path (str): path to extxyz-file
        db_path(str): path to sqlite database
    """
    with connect(db_path, use_lock_file=False) as conn:
        with open(extxyz_path) as f:
            for at in tqdm(read_xyz(f, index=slice(None)), "creating ase db"):
                data = {}
                if at.has("forces"):
                    data["forces"] = at.get_forces()
                data.update(at.info)
                conn.write(at, data=data)


def xyz_to_db(
    xyz_path,
    db_path,
    atomic_properties="Properties=species:S:1:pos:R:3",
    molecular_properties=[],
):
    """
    Convertes a xyz-file to an ase database.

    Args:
        xyz_path (str): path to the xyz file
        db_path(str): path to sqlite database
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # build temp file in extended xyz format
    extxyz_path = os.path.join(tempfile.mkdtemp(), "temp.extxyz")
    xyz_to_extxyz(xyz_path, extxyz_path, atomic_properties, molecular_properties)
    # build database from extended xyz
    extxyz_to_db(extxyz_path, db_path)


def generate_db(
    file_path,
    db_path,
    atomic_properties="Properties=species:S:1:pos:R:3",
    molecular_properties=[],
):
    """
    Convert a file with molecular information to an ase database. Currently
    supports .xyz and .extxyz.

    Args:
        file_path (str): path to the input file
        db_path(str): path to sqlite database
        atomic_properties (str): property-string
        molecular_properties (list): molecular properties contained in the
            comment line
    """
    # check if file extension is valid
    filename, file_extension = os.path.splitext(file_path)
    if file_extension not in [".xyz", ".extxyz"]:
        raise NotImplementedError(
            "{} is not a supported file " "extension!".format(file_extension)
        )
    # check if file has property string
    with open(file_path, "r") as file:
        _ = file.readline()
        comment_line = file.readline()
    # build database file
    if "Properties=" in comment_line:
        extxyz_to_db(file_path, db_path)
    else:
        xyz_to_db(file_path, db_path, atomic_properties, molecular_properties)


def parse_property_string(prop_str):
    """
    Generate valid property string for extended xyz files.
    (ref. https://libatoms.github.io/QUIP/io.html#extendedxyz)

    Args:
        prop_str (str): Valid property string, or appendix of property string

    Returns:
        valid property string
    """
    if prop_str.startswith("Properties="):
        return prop_str
    return "Properties=species:S:1:pos:R:3:" + prop_str
