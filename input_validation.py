def validate(file_path):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not file_path.lower().endswith(valid_extensions):
        return False

    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
        
        is_png=header.startswith(b'\x89PNG\r\n\x1a\n')
        is_jpg=header.startswith(b'\xff\xd8\xff')
        
        return is_png or is_jpg
    except IOError:
        return False
