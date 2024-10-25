import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_folder, output_folder, class_mapping):
    """
    Convierte archivos XML a formato YOLO.

    Args:
        xml_folder (str): Ruta a la carpeta que contiene archivos XML.
        output_folder (str): Ruta a la carpeta donde se guardarán los archivos TXT.
        class_mapping (dict): Mapeo de nombres de clases a índices.
    """
    # Asegurarse de que la carpeta de salida existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorrer todos los archivos XML en la carpeta de entrada
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Obtener dimensiones de la imagen
            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            yolo_data = []  # Lista para almacenar datos convertidos

            # Recorrer todos los objetos en el archivo XML
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in class_mapping:
                    class_id = class_mapping[class_name]
                    
                    # Obtener la bounding box
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    ymin = int(xmlbox.find('ymin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymax = int(xmlbox.find('ymax').text)

                    # Calcular las coordenadas normalizadas
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    # Agregar la línea con los datos normalizados al formato YOLO
                    yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Guardar en archivo .txt solo si se encontró algún objeto
            if yolo_data:
                base_filename = os.path.splitext(xml_file)[0]
                output_file_path = os.path.join(output_folder, f"{base_filename}.txt")
                with open(output_file_path, 'w') as f:
                    f.write("\n".join(yolo_data))
            else:
                print(f"Advertencia: No se encontraron objetos en {xml_file}")

# Configuración
xml_train_folder = 'dataset/labels/train'  # Ruta a las etiquetas de entrenamiento
xml_val_folder = 'dataset/labels/val'      # Ruta a las etiquetas de validación
output_train_folder = 'dataset/labels/train_yolo'  # Ruta a las etiquetas de entrenamiento en formato YOLO
output_val_folder = 'dataset/labels/val_yolo'      # Ruta a las etiquetas de validación en formato YOLO

# Mapeo de clases (ajusta según tus etiquetas)
class_mapping = {'label_0': 0}  # Aquí define las clases y sus índices

# Ejecutar conversión
convert_xml_to_yolo(xml_train_folder, output_train_folder, class_mapping)
convert_xml_to_yolo(xml_val_folder, output_val_folder, class_mapping)

print("Conversión completada.")
