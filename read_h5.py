import h5py

def inspect_hdf5_file(hdf5_filename):
    with h5py.File(hdf5_filename, 'r') as file:
        print("Inspecting HDF5 file:", hdf5_filename)
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    Attribute: {key}, Value: {val}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}, Type: {obj.dtype}")
        
        file.visititems(print_attrs)

# Example usage:
# Replace 'your_hdf5_file.h5' with the path to your actual HDF5 file
inspect_hdf5_file('au_train.h5')
