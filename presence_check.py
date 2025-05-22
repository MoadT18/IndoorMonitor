import serial
import datetime
import os

presence_file = '/home/moadt/indoor_monitor/presence.txt'
tmp_file = '/home/moadt/indoor_monitor/presence.tmp'

now = datetime.datetime.now()
minute = now.minute

# Reset presence een paar minuten NA het volle uur (bijv. minuut 2)
if minute == 2:
    try:
        with open(tmp_file, 'w') as f:
            f.write("OFF\n")
        os.replace(tmp_file, presence_file)  # Atomic safe move
    except Exception as e:
        print(f"🚨 Failed to reset presence file: {e}")

# Anders: check aanwezigheid
else:
    try:
        mmwave = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        mmwave.write(b'\xAA\x00\x00\x00\xAB')  # Wake-up command (optioneel)
        line = mmwave.readline()
        if line:
            decoded_line = line.decode('utf-8', errors='ignore').strip()

            if "ON" in decoded_line:
                try:
                    with open(tmp_file, 'w') as f:
                        f.write("ONN\n")
                    os.replace(tmp_file, presence_file)
                except Exception as e:
                    print(f"🚨 Failed to update presence file to ONN: {e}")
            # Else: do absolutely nothing, leave file as is
    except Exception as e:
        print(f"🚨 Failed to read mmWave sensor: {e}")

