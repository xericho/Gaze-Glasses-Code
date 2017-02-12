for i=1:900
    this_img=sprintf('%d.png',i);
    img=imread(this_img);
    thisfile=sprintf('object_%04d.jpeg',i);
    imwrite(img,thisfile,'jpeg');
    
end