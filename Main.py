import matplotlib.pylab as plt
from API import transfer_style


if __name__=="__main__":

    # Path of the pre-trained TF model 
    model_path = r"saved_model.pbtxt"
    # NOTE : Works only for '.jpg' and '.png' extensions,other formats may give error
    content_image_path = r"D:\new project titils and documentation pdf\Style Transfer App for Turning Photos into Artwork Using GANs Deep Learning\Neural-Style-Transfer-main\Imgs\content1.jpg"
    style_image_path = r"D:\new project titils and documentation pdf\Style Transfer App for Turning Photos into Artwork Using GANs Deep Learning\Neural-Style-Transfer-main\Imgs\content2.jpg"

    img = transfer_style(content_image_path,style_image_path,model_path)
    # Saving the generated image
    plt.imsave(r'D:\new project titils and documentation pdf\Style Transfer App for Turning Photos into Artwork Using GANs Deep Learning\Neural-Style-Transfer-main\stylized_image.jpeg',img)
    plt.imshow(img)
    plt.show()

