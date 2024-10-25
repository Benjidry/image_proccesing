import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_folder, output_folder, class_mapping):
    """
    Convierte archivos XML a formato YOLO para cada archivo en la carpeta XML.

    Args:
        xml_folder (str): Ruta de la carpeta que contiene archivos XML.
        output_folder (str): Ruta de la carpeta donde se guardarán los archivos TXT.
        class_mapping (dict): Mapeo de nombres de clases a índices.
    """
    # Asegurarse de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorrer todos los archivos XML en la carpeta de entrada
    for xml_file in sorted(os.listdir(xml_folder)):
        # Filtrar solo archivos que terminan en '.xml' y omitir archivos ocultos como '.DS_Store'
        if xml_file.endswith('.xml') and not xml_file.startswith('.'):
            xml_path = os.path.join(xml_folder, xml_file)
            print(f"Procesando archivo: {xml_file}")

            try:
                # Procesar el archivo XML
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Obtener dimensiones de la imagen
                img_width = int(root.find('size/width').text)
                img_height = int(root.find('size/height').text)
                print(f"Dimensiones: {img_width}x{img_height}")

                yolo_data = []  # Lista para almacenar los datos convertidos

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

                        print(f"Objeto encontrado: {class_name} - Box: ({xmin}, {ymin}), ({xmax}, {ymax})")

                        # Calcular las coordenadas normalizadas
                        x_center = (xmin + xmax) / 2 / img_width
                        y_center = (ymin + ymax) / 2 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        # Agregar la línea con los datos normalizados al formato YOLO
                        yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Guardar en archivo .txt solo si se encontró algún objeto
                if yolo_data:
                    base_filename = os.path.splitext(xml_file)[0]  # Elimina la extensión .xml
                    output_file_path = os.path.join(output_folder, f"{base_filename}.txt")
                    with open(output_file_path, 'w') as f:
                        f.write("\n".join(yolo_data))
                    print(f"Archivo generado: {output_file_path}")
                else:
                    print(f"Advertencia: No se encontraron objetos en {xml_file}")
            except ET.ParseError as e:
                print(f"Error al analizar el archivo XML: {xml_file} - {e}")
            except Exception as e:
                print(f"Error inesperado al procesar el archivo {xml_file}: {e}")
        else:
            print(f"Ignorado: {xml_file}")

# Configuración de rutas
xml_train_folder = 'dataset/labels/train'  # Ruta a las etiquetas de entrenamiento en XML
xml_val_folder = 'dataset/labels/val'      # Ruta a las etiquetas de validación en XML
output_train_folder = 'dataset/labels/train_yolo'  # Carpeta donde se guardarán los archivos .txt de entrenamiento
output_val_folder = 'dataset/labels/val_yolo'      # Carpeta donde se guardarán los archivos .txt de validación

# Mapeo de clases (ajusta según tus etiquetas)
class_mapping = {'face_azariel': 0}  # Mapea el nombre de tu clase al índice YOLO

# Ejecutar la conversión para el conjunto de entrenamiento y validación
convert_xml_to_yolo(xml_train_folder, output_train_folder, class_mapping)
convert_xml_to_yolo(xml_val_folder, output_val_folder, class_mapping)

print("Conversión completada para todos los archivos.")
