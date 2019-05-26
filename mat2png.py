#!/usr/bin/python

"""
MAT 2 extracting tool

This is intended to be used to support for example,
dozens of upscaling projects for Jedi Knight: Dark Forces II and
Jedi Knight: Mysteries of the Sith

"""

from struct import pack, unpack
from re import sub
import json
import logging
import glob
import numpy as np
import cv2
import sys
import os

########################
#  Logging configuration
########################
FORMAT = "%(asctime)-15s - [%(levelname)s] - %(message)s"
logging.basicConfig(filename="./"+sys.argv[0].split("/")[1].split(".")[0]+\
        ".log", format=FORMAT, \
        level=logging.INFO)

def colormap(name):
    """
    This function was created based on previous work on GORC project:
    https://github.com/jdmclark/gorc

    Initially, the maps used here were extracted from JK MotS.
    They can be changed to JKDF and JKDF2 as you see fit.
    For commodity, you can place them in the cmp directory,
    and refer to them calling the script.

    @param name: colormap name
    """
    name = name.split("/")[-1]
    with open("./cmp/"+name, "r") as cmap:
        data = cmap.read()
        cmap.close()
    #  Offset 0x00
    ftype = "".join(c for c in unpack("<cccc", data[0:4]))
    version = unichr(unpack("<I", data[4:8])[0]) #  0x20, 0x1e 
    file_type = "".join(ftype+version)
    transparent_bool = True if unpack("<I", data[8:12])[0] is not 0 else\
            False #  != 0 is True
    color_tint = unpack("<fff", data[12:24])[0] #  tint
    p_start = 24 + (4*10) #  start palette offset
    raw_palette = []
    nexp = p_start + 1
    for p in range(256*3):
        raw_palette.append(unpack("<B", data[p_start:nexp])[0]) #  color palette
        nexp += 1
        p_start += 1
    palette_array = np.ndarray((16,16, 3), dtype=np.uint8)
    pos = 0
    #  strat write color palette
    for y in range(16):
        for x in range(16):
            bgr = (raw_palette[pos+2], raw_palette[pos+1], raw_palette[pos])
            palette_array[y, x] = bgr 
            pos += 3 #  shift 3 positions from last 
    #palette = cv2.cvtColor(palette_array, cv2.COLOR_BGR2RGB)
    pos = 0
    palette = palette_array
    cv2.imwrite("./palettes/"+name+"-palette.png", palette)
    transp_array = np.ndarray((16,16, 4), dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            bgra = (raw_palette[pos+2], raw_palette[pos+1], raw_palette[pos], 255)
            transp_array[y, x] = bgra 
            pos += 3 #  shift 3 positions from last
    transparency_table = transp_array
    lighting_table = transparency_table #  not sure what to do with lighting
    #  table. Copying from transparency table in order not to mess
    #  with function returning values structure for now.
    if version == u'\x1e':
        return (palette, lighting_table, transparency_table)
    else:
        print "ERROR: unable to identify colormap version / type."
        sys.exit(127)


def mat_hdr(data):
    """
    This function handles the MAT header parsing.
    @param data: MAT data as a bytearray.
    """
    #  Offset 0x00
    ftype = "".join(c for c in unpack("<cccc", data[0:4]))
    version = unichr(unpack("<I", data[4:8])[0]) #  0x20, 0x32 for 16-bit
    file_type = "".join(ftype+version)
    mat_type = unpack("<I", data[8:12])[0] #  0 = color; 1 = ?; 2 = texture
    if mat_type is 0:
        mat_type = "color"
    if mat_type is 1:
        mat_type = "unknown"
    if mat_type is 2:
        mat_type = "texture"
    record_count = unpack("<I", data[12:16])[0]    
    texture_count = unpack("<I", data[16:20])[0]
    #  skip Transparency
    """
    Based on:
    http://www.jkhub.net/library/index.php?title=Reference:Material

    It is not accurate to determine transparency from the MAT header.
    So, we skip it.
    """
    bitdepth = unpack("<I", data[24:28])[0] #  8 (converted), 16 or 32
    """
    BGR bits
    B = 0, 5, 8
    G = 0, 6 (16-bit 565), 5 (16-bit 1555 ARGB), 8
    R = 0, 5, 8
    """
    blue = unpack("<I", data[28:32])[0]
    green = unpack("<I", data[32:36])[0]
    red = unpack("<I", data[36:40])[0]
    """
    Not considering ARGB1555 mode for now
    as we are not defining material transparency here.
    """
    if bitdepth is 16: #  convert to RGB565
        red = (red & 0xf8) << 8
        green = (green & 0xfc) << 3
        blue = blue >> 3
        red = red << 11
        green = green << 5
    #  Shift Right
    red = red >> 3
    green = green >> 2
    blue = blue >> 3
    #  End offset 0x4b
    return (file_type, mat_type, record_count, texture_count, bitdepth, \
            red, green, blue)


def mat_record_hdr(data):
    """
    This function determines the material record type
    
    @param data: MAT data as a bytearray.
    """
    #  Offset 0x4c
    offset = 76
    entries = unpack("<I", data[12:16])[0]
    record_list = []
    for i in range(entries):
        record_number = i
        transparent_color = unpack("<I", data[offset+4:offset+8])[0]
        if mat_hdr(data)[1] is 0:
            break
            return 1
        else:
            mat_record = (record_number, transparent_color, offset)
            record_list.append(mat_record)
            offset += 40 
    #  ^ if 16-bit or 32-bit, RGB. If 8-bit, palette offset
    #  Add 32 padding bytes
    #  End offset = variable length
    return record_list


def texture_hdr(data, offset):
    """
    This function returns the material dimensions,
    transparency, mipmap count and also where the bitmap data
    starting offset
    
    @param data: MAT data as a bytearray.
    @param offset: MAT color header offset.
    """
    #  Offset variable length; add 40 bytes
    offset += 40
    width = unpack("<I", data[offset:offset+4])[0]
    height = unpack("<I", data[offset+4:offset+8])[0]
    transparent_bool = unpack("<I", data[offset+8:offset+12])[0]
    #  DWORD of unknown fields?
    mipmap_count = unpack("<I", data[offset+20:offset+24])[0]
    # End offset variable length
    bitmap_offset = offset + 24
    return (width, height, transparent_bool, mipmap_count, bitmap_offset)


def mat_color_record_hdr(data):
    #  Offset for color type is always 0x4c
    offset = 76
    record_type = unpack("<I", data[offset:offset+4])[0]
    color_num = unpack("<I", data[offset+4:offset+8])[0]
    if record_type is 0 and color_num:
        return (record_type, color_num)
    else:
        return 1


def generate_image(x, y, data, cmapname, transparent_bool=False):
    """
    This function returns the image payload data to be written to disk

    @param x: col size
    @param y: row size
    @param data: bitmap data
    @param cmapname: colormap name
    @param transparent_bool: boolean
    """

    pos = 0
    palette, lighting_table, transparency_table = colormap(cmapname)
    if transparent_bool is True:
        image = np.zeros((y, x, 4), dtype=np.uint8)
    else:
        image = np.zeros((y, x, 3), dtype=np.uint8)
    for i in range(y):
        for j in range(x):
            if transparent_bool is True:
                b = data[pos]
                g = data[pos]
                r = data[pos]
                a = data[pos]
                image[i, j] = (b, g, r, a)
                pos += 1
            else:
                b = data[pos]
                g = data[pos]
                r = data[pos]
                image[i, j] = (b, g, r)
                pos += 1
    if transparent_bool is True:
        image = cv2.LUT(image, transparency_table)
    else:
        image = cv2.LUT(image, palette)
    return image


def write_image(pathname, x, y, data, cmapname, cel_count, mipmap_count, \
        transparent_bool=False):
    """
    This function ultimately writes the image data to disk and returns
    a JSON payload, compatible to JKGFX mod script file.

    @param pathname: pathname where the file(s) will be written
    @param x: col size
    @param y: row size
    @param cel_count: number of cels found in the material header
    @param mipmap_count: mipmap count
    @transparent_bool: boolean
    """
    #  Start JSON data
    metadata = []
    # init index
    pos = 0
    size = (x * y)+((2**3)*3)
    cx, cy = x,  y 
    orig_data = data
    for i in range(cel_count):
        nx, ny = x, y
        #metadata.append({"replaces": []})
        for m in range(mipmap_count):
            pname = pathname
            if "3do" in pname:
                fname = "jknup.3do.mat."+pname.split("/")[-1].split(".")[0]\
                        +"-"+str(i)+"-"+str(m)+".png"
                mat_name = "3do/mat/"+pname.split("/")[-1].split(".")[0]
            else:
                fname = "jknup.mat."+pname.split("/")[-1].split(".")[0]\
                        +"-"+str(i)+"-"+str(m)+".png"
                mat_name = "mat/"+pname.split("/")[-1].split(".")[0]
            pname = "/".join([c for c in pname.split("/")[:-1]])
            try:
                img = generate_image(x, y, data[0:], cmapname, transparent_bool)
                img = cv2.resize(img, (nx, ny))
            except:
                logging.warn("cel %s has a different dimension." % i)
                logging.warn("Assuming mipmap behavior:")
                logging.warn("Resizing cel %s to: %sx%s" % (i, cy, cx))
                img = generate_image(x, y, orig_data, cmapname, transparent_bool)
                img = cv2.resize(img, (cx, cy))
            #  Reset compression level to 3. (default)
            cv2.imwrite(pname+"/"+fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            ny = ny >> 1
            nx = nx >> 1
        #  mipmap is not properly supported by JKGFX Mod, so we 
        #  retain the original image name instead. (quick dirty fix below)
        #  OBS: all the mipmaps are extracted regardless, they're
        #  bypassed in the JSON.
        fname = sub(r"-(\d+)-\d+", r"-\g<1>-0", fname)
        metadata.append({"albedo_map": fname})
        metadata[pos].update({"name": sub(r"\.png", "", fname)})
        if cel_count is 1:
            metadata[pos].update({"replaces":  mat_name+".mat"})
        else:
            metadata[pos].update({"replaces": [[mat_name+".mat", i]]})
        pos += 1
        cx = cx >> 1
        cy = cy >> 1
        data = data[size:]
    return metadata


def usage():
    print "ERROR: missing argument(s)\n\n\
%s <MAT_FILE_OR_PATH> <DESTINATION_PATHNAME> <COLORMAP>\n" \
% sys.argv[0]
    logging.error("Missing argument(s)\n\
           %s <MAT_FILE_OR_PATH> <DESTINATION_PATHNAME> <COLORMAP>" % sys.argv[0])
    sys.exit(1)

def main():
    if len(sys.argv) < 4:
        usage()
    cmap_name = sys.argv[3]
    metadata = {"materials": []}
    if os.path.isfile(sys.argv[1]) and os.stat(sys.argv[1]).st_size == 0:
        print "ERROR: %s is empty" % sys.argv[1]
        logging.error("%s is empty" % sys.argv[1])
        sys.exit(1)
    elif os.path.isfile(sys.argv[1]) and os.stat(sys.argv[1]).st_size > 0:
        name = sys.argv[1].split("/")[-1].split(".")[0]
        path = sub(r"\/$", "", sys.argv[2])
        pathname = "".join(path+"/"+name)
        with open(sys.argv[1], 'r') as mat:
            material = bytearray(mat.read())
            mat.close()
        mat_type = mat_hdr(material)[1]
        file_type = mat_hdr(material)[0]
        if file_type == 'MAT 2':
            bitdepth = mat_hdr(material)[4]
            if mat_type == "color":
                bitdepth = mat_hdr(material)[4]
                record_type, color_num = mat_color_record_hdr(material)
                logging.info("Nothing to do with file: %s ..." % sys.argv[1])
                logging.info("File type and version: %s" % file_type)
                logging.info("color number: %s" % color_num)
                logging.info("bitdepth: %s" % bitdepth)
                logging.info("Given file is not a texture")
            if mat_type == "texture":
                print "Extracting: %s ..." % sys.argv[1]
                logging.info("Extracting: %s ..." % sys.argv[1])
                logging.info("File type and version: %s" % file_type)
                logging.info("mat type: %s" % mat_type)
                logging.info("bitdepth: %s" % bitdepth)
                offset = mat_record_hdr(material)[-1][-1]
                w, h, transparent_bool, mipmap_count, bitmap_offset \
                = texture_hdr(material, offset)
                transparent_bool = False if transparent_bool is 0 else True
                logging.info("dimension: %sx%s" % (h, w))
                logging.info("transparent? %s" % transparent_bool)
                logging.info("mipmap count: %s" % mipmap_count)
                for t in mat_record_hdr(material):
                    tex_number = t[0]
                    transparent_color = t[1]
                    logging.info("cel: %s - palette color index: %s" \
                    % (tex_number, transparent_color))
                total_number_of_cels = len(mat_record_hdr(material))
                bitmap_data = material[bitmap_offset:]
                mat_list = write_image(pathname, w, h, bitmap_data, cmap_name,\
                total_number_of_cels, mipmap_count, transparent_bool)
                for m in mat_list:
                    metadata["materials"].append(m)
                metadata.update({"description":\
"This material pack contains upscaled textures using ERSGAN and a custom model. (shout to TreeMarmot for the model)"})
                with open(path+"/metadata.json", "w") as f:
                    f.write(json.dumps(metadata, indent=4, sort_keys=True))
                    f.close()
            else:
                logging.error("Invalid material type")
                sys.exit(1)
    elif os.path.isdir(sys.argv[1]):
        path = sub(r"\/$", "", sys.argv[2])
        for f in list(glob.glob(sys.argv[1]+'/*.mat')):
            name = f.split("/")[-1].split(".")[0]
            pathname = "".join(path+"/"+name)
            if os.stat(f).st_size == 0:
                print "ERROR: %s is empty" % f
                logging.error("%s is empty" % f)
                continue
            else:
                with open(f, 'r') as mat:
                    material = bytearray(mat.read())
                    mat.close()
                mat_type = mat_hdr(material)[1]
                file_type = mat_hdr(material)[0]
                if file_type == 'MAT 2':
                    bitdepth = mat_hdr(material)[4]
                    if mat_type == "color":
                        bitdepth = mat_hdr(material)[4]
                        record_type, color_num = mat_color_record_hdr(material)
                        logging.info("Nothing to do with file: %s ..." % sys.argv[1])
                        logging.info("File type and version: %s" % file_type)
                        logging.info("color number: %s" % color_num)
                        logging.info("bitdepth: %s" % bitdepth)
                        logging.info("Given file is not a texture")
                    if mat_type == "texture":
                        print "Extracting: %s ..." % f
                        logging.info("Extracting: %s ..." % f)
                        logging.info("File type and version: %s" % file_type)
                        logging.info("mat type: %s" % mat_type)
                        logging.info("bitdepth: %s" % bitdepth)
                        offset = mat_record_hdr(material)[-1][-1]
                        w, h, transparent_bool, mipmap_count, bitmap_offset \
                        = texture_hdr(material, offset)
                        transparent_bool = False if transparent_bool is 0 else True
                        logging.info("dimension: %sx%s" % (h, w))
                        logging.info("transparent? %s" % transparent_bool)
                        logging.info("mipmap count: %s" % mipmap_count)
                        for t in mat_record_hdr(material):
                            tex_number = t[0]
                            transparent_color = t[1]
                            logging.info("cel: %s - palette color index: %s" \
                            % (tex_number, transparent_color))
                        total_number_of_cels = len(mat_record_hdr(material))
                        bitmap_data = material[bitmap_offset:]
                        mat_list = write_image(pathname, w, h, bitmap_data, cmap_name,\
                        total_number_of_cels, mipmap_count, transparent_bool) 
                        for m in mat_list:
                            metadata["materials"].append(m)
                        metadata.update({"description":\
"This material pack contains upscaled textures using ERSGAN and a custom model. (shout to TreeMarmot for the model)"})
                    else:
                        logging.error("Invalid material type")
                        continue
        with open(path+"/metadata.json", "w+") as mf:
            mf.write(json.dumps(metadata, indent=4, sort_keys=True))
            mf.close()
    else:
        sys.exit(1)
        print "ERROR: unable to determine fd type."
if __name__ == "__main__":
    main()
