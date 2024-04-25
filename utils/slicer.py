import SimpleITK as sitk
def ct_to_slices(img_path):
    slices = []
    image = sitk.ReadImage(img_path)
    # print(image)
    image_np = sitk.GetArrayFromImage(image)
    for i in range(image_np.shape[0]):
        
        slices.append(image_np[i,:,:])
    # print(image_np.shape)
    return slices

