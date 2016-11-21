function save_pair( imgdir, maskdir, db, HOMEIMAGES )

car_list_path = strcat(HOMEIMAGES, '/car_list.txt');
car_list_fid = fopen(car_list_path, 'w');

for i = 1:length(db)
    
    imgfile_this = db(i).annotation;
    obj_this = imgfile_this.object;
    folder_this = imgfile_this.folder;
    file_this = imgfile_this.filename;
    [~, name, ~] = fileparts(file_this);
    img_this = strcat(HOMEIMAGES, '/' , folder_this, '/', file_this);
    A = imread(img_this);
    
    [newannotation, newimg, ~, ~, ~, ~] = ...
        LMcookimage(imgfile_this, A, 'maximagesize', [256 256], 'impad', 0);

    folder_dir = strcat(maskdir, folder_this);
    
    if ~exist(folder_dir, 'dir')
        mkdir(folder_dir)         
    end
    
    % all masks
    newannotation.imagesize.nrows = size(newimg, 1);
    newannotation.imagesize.ncols = size(newimg, 2);
    [mask, ~] = LMobjectmask(newannotation, [256 256]);
    
    num_obj = length(newannotation.object);
    for ii = 1:num_obj
        
        this_mask = mask(:,:,ii);
        if (sum(this_mask(:))<eps)
            continue;
        end
        
        sub_obj = obj_this(ii);
        poly_x = sub_obj.polygon.x;
        poly_y = sub_obj.polygon.y;
        
        if length(poly_x) < 6
            continue;
        end
        
        xrange = max(poly_x) - min(poly_x);
        yrange = max(poly_y) - min(poly_y);
        
        if (xrange <= 15 || yrange <= 15)
            continue;
        end
        
        if isfield(sub_obj,'id')
            id_this = sub_obj.id;
        else
            id_this = 100+ii;
        end
        
        filename = strcat(name, '_', num2str(id_this));
        
        img_folder_dir = strcat(imgdir, folder_this);
        if (~exist(img_folder_dir, 'dir'))
            mkdir(img_folder_dir)
        end
        img_dir = strcat(imgdir, folder_this, '/', filename);
        imwrite(newimg, img_dir, 'PNG');
        
        mask_folder_dir = strcat(maskdir, folder_this);
        if (~exist(mask_folder_dir, 'dir'))
            mkdir(mask_folder_dir)
        end
        mask_dir = strcat(maskdir, folder_this, '/', filename);
        imwrite(mask(:,:,ii), mask_dir,'PNG');
        
        fprintf(car_list_fid,'%s %s\n', img_dir, mask_dir);
       
       % imshow(colorSegments(mask(:,:,ii)))
       
    end
    fprintf('processing %d\n',i)
end

fclose(car_list_fid);
fprintf('Done Saving Masks into %s\n', maskdir);

end

