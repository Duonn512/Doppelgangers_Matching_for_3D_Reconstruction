class VisualDisambiguationModel(nn.Module):
    def __init__(self, backbone, depth_model, resnet_out_channels=512):
        super().__init__()
        self.backbone = backbone
        self.depth_model = depth_model
        self.resnet = ResNetBlocks(1, 64, resnet_out_channels)  # Matching input channels
        self.classifier = nn.Sequential(
            nn.Linear(resnet_out_channels * 24 * 32 * 2, 256),  # Adjust the input to the correct size
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        # Forward pass through the backbone for both images
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # Estimate depth for both images
        depth1 = self.depth_model.whole_inference(feat1.unsqueeze(0), img_meta=None, rescale=True)
        depth2 = self.depth_model.whole_inference(feat2.unsqueeze(0), img_meta=None, rescale=True)

        # Apply depth as a mask (assuming depth is a single channel)
        mask1 = depth1.squeeze().cpu().detach().numpy()
        mask2 = depth2.squeeze().cpu().detach().numpy()

        # Apply masks to features
        feat1 = feat1 * torch.tensor(mask1).unsqueeze(0).to(feat1.device)
        feat2 = feat2 * torch.tensor(mask2).unsqueeze(0).to(feat2.device)

        # Flatten the features and concatenate
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        combined_features = torch.cat((feat1, feat2), dim=1)

        return self.classifier(combined_features)