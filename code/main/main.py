def train(config):
    # Local weight and sample directories
    weight_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'weights')
    sample_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'samples')
    run_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'runs')

    # Google Drive directory for saving weights
    drive_weight_dir = '/content/drive/My Drive/SIZE_DOES_MATTER/checkpoints'
    os.makedirs(drive_weight_dir, exist_ok=True)

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    dataset = TryonDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                            shuffle=True, num_workers=config['TRAINING_CONFIG']['NUM_WORKER'])
    print('Size of the dataset: %d, dataloader: %d' % (len(dataset), len(dataloader)))

    model = COTTON(config).cuda().train()
    discriminator = FashionOn_MultiD(shape=config['TRAINING_CONFIG']['RESOLUTION']).cuda().train()

    # Loss functions and optimizers
    criterion_VGG = FashionOn_VGGLoss()
    criterion_L1 = nn.L1Loss()
    criterionBCE = nn.BCELoss()
    focal_weight = [2, 3, 3, 3, 10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
    criterion_focal = focal_loss(alpha=focal_weight, gamma=2)

    optimizer_G = torch.optim.Adam(model.parameters(), lr=config['TRAINING_CONFIG']['LR'],
                                   betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['TRAINING_CONFIG']['LR'],
                                   betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))

    scheduler = None
    if config['TRAINING_CONFIG']['SCHEDULER'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,
                                                               T_max=config['TRAINING_CONFIG']['T_MAX'],
                                                               eta_min=config['TRAINING_CONFIG']['ETA_MIN'])

    weight_path = os.path.join(weight_dir, '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))
    best_model_path = os.path.join(weight_dir, 'best_model.pkl')
    best_drive_path = os.path.join(drive_weight_dir, 'best_model.pkl')

    # Check for checkpoints on Google Drive first
    start_epoch = 0
    best_score = float('inf')
    checkpoint_found = False
    if os.path.isfile(best_drive_path):
        print("=> Found checkpoint on Google Drive: '{}'".format(best_drive_path))
        checkpoint = torch.load(best_drive_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint from Google Drive (epoch {})".format(checkpoint['epoch']))
        best_score = checkpoint.get('best_score', best_score)
        checkpoint_found = True
    elif os.path.isfile(weight_path):
        print("=> Found local checkpoint: '{}'".format(weight_path))
        checkpoint = torch.load(weight_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded local checkpoint (epoch {})".format(checkpoint['epoch']))
        best_score = checkpoint.get('best_score', best_score)
        checkpoint_found = True

    if not checkpoint_found:
        print("=> No checkpoint found. Starting training from scratch.")

    board = SummaryWriter(run_dir)

    for epoch in range(start_epoch, config['TRAINING_CONFIG']['EPOCH']):
        print("epoch: " + str(epoch + 1))
        epoch_loss = 0

        for step, batch in enumerate(tqdm(dataloader)):
            total_step = epoch * len(dataloader) + step

            # Training logic
            human_masked = batch['human_masked'].cuda()
            human_pose = batch['human_pose'].cuda()
            human_parse_masked = batch['human_parse_masked'].cuda()
            human_img = batch['human_img'].cuda()
            human_parse_label = batch['human_parse_label'].cuda()
            c_aux = batch['c_aux_warped'].cuda()
            c_torso = batch['c_torso_warped'].cuda()

            c_img = torch.cat([c_torso, c_aux], dim=1)
            parsing_pred, parsing_pred_hard, tryon_img_fake = model(c_img, human_parse_masked, human_masked, human_pose)

            # Calculate losses
            loss_focal = criterion_focal(parsing_pred, human_parse_label)
            loss_content_gen, loss_style_gen = criterion_VGG(tryon_img_fake, human_img)
            loss_vgg = config['TRAINING_CONFIG']['LAMBDA_VGG_STYLE'] * loss_style_gen + config['TRAINING_CONFIG']['LAMBDA_VGG_CONTENT'] * loss_content_gen
            loss_global_L1 = criterion_L1(tryon_img_fake, human_img)

            loss_G = config['TRAINING_CONFIG']['LAMBDA_L1'] * loss_global_L1 + \
                     config['TRAINING_CONFIG']['LAMBDA_VGG'] * loss_vgg + \
                     config['TRAINING_CONFIG']['LAMBDA_FOCAL'] * loss_focal

            # Backpropagation and optimization
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            epoch_loss += loss_G.item()  # Accumulate generator loss

        # Save checkpoint
        save_info = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'state_dict_D': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'best_score': best_score,
        }
        if scheduler:
            scheduler.step()
            save_info['scheduler'] = scheduler.state_dict()

        torch.save(save_info, weight_path)  # Save latest checkpoint locally

        # Save best model
        avg_epoch_loss = epoch_loss / len(dataloader)
        if avg_epoch_loss < best_score:
            best_score = avg_epoch_loss
            save_info['best_score'] = best_score
            torch.save(save_info, best_model_path)  # Save best model locally
            torch.save(save_info, best_drive_path)  # Save best model to Google Drive
            print(f"Best model updated with loss {best_score:.4f} and saved to Google Drive.")
