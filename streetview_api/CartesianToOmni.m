function OutputImgs = CartesianToOmni(dir, seqname)
    %% Plot Spherical Point Cloud with Texture Mapping
    % Generate a sphere consisting of 2400-by-2400 faces.

    numFaces = 2400;
    [x,y,z] = sphere(numFaces);
    %% 
    % Plot the sphere using the default color map.
    %%
    %figure;
    %pcshow([x(:),y(:),z(:)]);
    %title('Sphere with Default Color Map');
    %xlabel('X');
    %ylabel('Y');
    %zlabel('Z');
    
    I = im2double(imread(dir));
    %% 
    % Resize and flip the image for mapping the coordinates.
    %%
    J = flipud(imresize(I,size(x)));
    %% 
    % Plot the sphere with the color texture.
    %%

    %title('Sphere with Color Texture');
    %xlabel('X');
    %ylabel('Y');
    %zlabel('Z');
    pcshow([x(:),y(:),z(:)],reshape(J,[],3));
    view(0,90);
    axis off;
    grid off;
    %export_fig test.png -transparent
    
    parsedname = strsplit(dir, '/')
    filenamecellarray = parsedname(end)
    filename = filenamecellarray{1}
    
    temp = sprintf('../img_files/%s/', seqname)
    savedir = strcat(temp, filename)
    set(gcf, 'Color','black')
    export_fig(savedir, '-jpg', '-m2')
    %% 
    % _Copyright 2015 The MathWorks, Inc._
