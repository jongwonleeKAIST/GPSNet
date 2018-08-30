function OutputImgs = CartesianToOmni(dir, seqname)
    %% Plot Spherical Point Cloud with Texture Mapping
    % Generate a sphere consisting of 2400-by-2400 faces.

    numFaces = 3000;
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
    filename = filename(1:end-4);
    
    temp = sprintf('../img_files/%s/', seqname)
    savedir = strcat(temp, filename)
    set(gcf, 'Color','black', 'PaperPosition', [0 0 224/150 224/150])
    %set(gcf, 'Color','black', 'PaperUnits','inches')
    export_fig(savedir, '-dpng', '-transparent', '-r300')
    
    %fig = gcf;
    %fig.PaperUnits = 'inches';
    %fig.PaperPosition = [0 0 224/150 224/150];
    %print('SameAxisLimits','-dpng','-r0')
    
    %% 
    % _Copyright 2015 The MathWorks, Inc._
