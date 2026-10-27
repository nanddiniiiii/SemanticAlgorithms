"""Small experimental SQ-GAN source packet: two bitmaps, VQ IDs, and CRC32.

The packet format is ours, not the paper's entropy coder or a radio protocol.
CRC detects corruption; it provides no error correction or delivery guarantee.
"""
import struct
import zlib

HEADER=struct.Struct('!4sHHHH')
TOKENS=512
CODEBOOK=1024


def pack_codes(values):
    accumulator=0
    count=0
    output=bytearray()
    for value in values:
        value=int(value)
        if not 0 <= value < CODEBOOK:
            raise ValueError('VQ index outside the 1024-entry codebook')
        accumulator=(accumulator<<10)|value
        count+=10
        while count>=8:
            count-=8
            output.append((accumulator>>count)&255)
    if count:
        output.append((accumulator<<(8-count))&255)
    return bytes(output)


def unpack_codes(data,count):
    if len(data)!=(count*10+7)//8:
        raise ValueError('Invalid packed-code length')
    value=int.from_bytes(data,'big')
    padding=len(data)*8-count*10
    if padding and value&((1<<padding)-1):
        raise ValueError('Nonzero padding bits')
    return [(value>>(len(data)*8-10*(i+1)))&1023 for i in range(count)]


def encode_packet(image_positions,image_codes,semantic_positions,semantic_codes):
    sections=[]
    counts=[]
    for positions,codes in [(image_positions,image_codes),(semantic_positions,semantic_codes)]:
        if len(positions)!=len(codes) or not 0<len(positions)<=TOKENS:
            raise ValueError('Invalid number of selected tokens')
        if len(set(positions))!=len(positions) or any(not 0<=int(p)<TOKENS for p in positions):
            raise ValueError('Invalid or repeated token positions')
        pairs=sorted(zip(positions,codes))
        mask=bytearray(TOKENS//8)
        for position,_ in pairs:
            position=int(position)
            mask[position//8]|=1<<(7-position%8)
        sections.append(bytes(mask)+pack_codes([code for _,code in pairs]))
        counts.append(len(pairs))
    payload=HEADER.pack(b'SQ01',256,512,*counts)+b''.join(sections)
    return payload+struct.pack('!I',zlib.crc32(payload))


def decode_packet(packet):
    if not 144<=len(packet)<=1424:
        raise ValueError('SQ packet length is outside bounds')
    payload=packet[:-4]
    if zlib.crc32(payload)!=struct.unpack('!I',packet[-4:])[0]:
        raise ValueError('SQ packet CRC failure; delivery failed')
    magic,height,width,image_count,semantic_count=HEADER.unpack(payload[:HEADER.size])
    if magic!=b'SQ01' or (height,width)!=(256,512):
        raise ValueError('Unsupported SQ packet version or dimensions')
    expected=HEADER.size+128+(image_count*10+7)//8+(semantic_count*10+7)//8
    if len(payload)!=expected:
        raise ValueError('SQ packet field lengths disagree')
    offset=HEADER.size
    sections=[]
    for count in [image_count,semantic_count]:
        if not 0<count<=TOKENS:
            raise ValueError('Invalid SQ token count')
        mask=payload[offset:offset+64];offset+=64
        positions=[i for i in range(TOKENS) if mask[i//8]&(1<<(7-i%8))]
        if len(positions)!=count:
            raise ValueError('Bitmap and token count disagree')
        length=(count*10+7)//8
        codes=unpack_codes(payload[offset:offset+length],count);offset+=length
        sections.append((positions,codes))
    return sections
