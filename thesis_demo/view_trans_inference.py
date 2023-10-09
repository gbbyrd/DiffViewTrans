def run_translation_rgb_depth(model, 
                              opt,
                              ):
    """Runs inference using depth - instance segmentation images and model.
    """

    # create dataloader
    dataset = RGBDepthDatasetBase(opt.sample_data_folder) # for depth instance sampling
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    if opt.vanilla_sample:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        model.num_timesteps = opt.custom_steps
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # sample batches from the dataloader and run inference
    img_count = 0
    for batch in dataloader:
        z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
        

        samples, intermediates = sampler.sample(200, opt.batch_size, 
                                                shape=(3, 64, 64), 
                                                conditioning=c, verbose=False,
                                                translation_label=translation_label, 
                                                eta=1.0)

        # decode from latent with pretrained autoencoder
        imgs = model.decode_first_stage(samples)
        
        # save the translated imgs
        for i in range(opt.batch_size):
            # process translated img
            to_view = torch.clamp(imgs[i], -1., 1.)
            to_view = to_view.cpu().numpy()
            to_view = np.transpose(to_view, (1, 2, 0))
            to_view = denormalize(to_view)

            # get ground truth and conditioning imgs
            ground_truth = batch['to'][i].numpy()
            from_view = batch['from'][i].numpy()

            # denormalize data images loaded from dataloader
            ground_truth = denormalize(ground_truth)
            from_view = denormalize(from_view)

            # uncomment below to display images for 5 seconds

            # cv2.imshow('ground_truth', ground_truth)
            # cv2.imshow('aerial', aerial_view)
            # cv2.imshow('translated', trans_img)
            # cv2.waitKey(5000)

            # concatenate together
            full_img = np.concatenate([from_view, ground_truth, to_view], axis=1)

            # break each 4 channel image into depth and rgb images
            full_rgb_img = np.concatenate([from_view[:, :, :3], ground_truth[:, :, :3], to_view[:, :, :3]], axis=1)
            full_depth_img = np.concatenate([from_view[:, :, -1], ground_truth[:, :, -1], to_view[:, :, -1]], axis=1)

            # save the imgs
            # cv2.imwrite(os.path.join(opt.save_dir, f'processed_sample_{img_count}.png',),
            #             processed_data)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}_rgb.png'),
                        full_rgb_img)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}_depth.png'),
                        full_depth_img)
            img_count += 1
            
            # break out if number of samples reached
            if img_count >= opt.n_samples:
                break
        
        if img_count >= opt.n_samples:
                break