import os

def setup_flightgear_xml(port=5550, rate=60) -> str:
    """
    FlightGear canlı UDP yayını için gereken XML dosyasını otomatik oluşturur 
    ve JSBSim'in çökmemesi için dosyanın TAM yolunu (Absolute Path) döndürür.
    """
    xml_content = f"""<?xml version="1.0"?>
<output name="127.0.0.1" type="FLIGHTGEAR" port="{port}" protocol="udp" rate="{rate}">
    <position> ON </position>
    <attitude> ON </attitude>
    <velocities> ON </velocities>
</output>
"""
    file_name = "fg_output.xml"
    with open(file_name, "w") as f:
        f.write(xml_content)
    
    xml_absolute_path = os.path.abspath(file_name)
    print(f"FlightGear XML Köprüsü Hazır: {xml_absolute_path}")
    return xml_absolute_path