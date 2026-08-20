from radcnnfuse import RadCNNFuse

rf = RadCNNFuse()

df = rf.transform_dataset(
    image_folder="my_images/"
)
