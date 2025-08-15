from datetime import datetime

def make_metadata_file(
        mrc_filepath: str, 
        pdb_filepath: str,
        added_b: float,
        upsampling: int,
        scaled_b: float, 
        pixel_size: float, 
        volume_size: tuple[int, int, int], 
        centered: bool, 
        dose_start: float, 
        dose_end: float, 
        voltage: float,
) -> None:
    """
    Write the metadata file with simulator parameters in the same place as the final volume.
    """
    metadata_filepath = mrc_filepath.replace(".mrc", "_sim_parameters.txt")
    with open(metadata_filepath, 'w') as f:
        f.write("Simulator parameters:\n")
        f.write(f"Time simulated: {datetime.now()}\n")
        f.write(f"PDB filepath: {pdb_filepath}\n")
        f.write(f"Output: {mrc_filepath}\n")
        f.write(f"Voltage: {voltage:.2f}\n")
        f.write(f"Dose start: {dose_start:.2f}\n")
        f.write(f"Dose end: {dose_end:.2f}\n")
        f.write(f"Upsampling: {upsampling}\n")
        f.write(f"Volume size: {volume_size}\n")
        f.write(f"Centered atoms?: {centered}\n")
        f.write(f"Scaled b-factor: {scaled_b:.2f}\n")
        f.write(f"Added b-factor: {added_b:.2f}\n")
        f.write(f"Pixel size: {pixel_size:.4f}\n")