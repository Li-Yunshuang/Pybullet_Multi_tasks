import imageio


if __name__ == '__main__':

    


    

    gif_images = []
    for i in range(143):
        path = './1/'+str(1+i*4)+'.png'


        gif_images.append(imageio.imread(path))
    
    name = 'ab.gif'
        
    imageio.mimsave(name,gif_images,fps=60)
    #imageio.mimsave("test33.gif", gif_images, 'GIF', duration = 1/240)
