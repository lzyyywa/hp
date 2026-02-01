def _init_hyperbolic_modules(self):
        print("[CustomCLIP] Applying Identity Initialization (Preserving CLIP Alignment)...")
        # 遍历所有投影层
        for m in [self.c2c_OE1, self.c2c_VE1, self.c2c_text_v, self.c2c_text_o]:
            if isinstance(m, nn.Module):
                for sub_m in m.modules():
                    # 1. 线性层：初始化为单位矩阵 (Identity)
                    if isinstance(sub_m, nn.Linear):
                        if sub_m.in_features == sub_m.out_features:
                            nn.init.eye_(sub_m.weight) # <--- 核心修改：变成 Identity
                            # 加入极微小的噪声打破完美对称，利于梯度流动
                            sub_m.weight.data.add_(torch.randn_like(sub_m.weight) * 0.001)
                        else:
                            # 维度不同时无法做 Identity，退回正交
                            nn.init.orthogonal_(sub_m.weight, gain=1.0)
                        
                        if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0.0)
                    
                    # 2. 卷积层 (MLP_ST)：初始化为 Dirac (相当于 Identity)
                    elif isinstance(sub_m, nn.Conv1d):
                         nn.init.dirac_(sub_m.weight) # <--- 核心修改：卷积的 Identity 是 Dirac 分布
                         if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0.0)
                            
                    # 3. LayerNorm：标准初始化
                    elif isinstance(sub_m, nn.LayerNorm):
                        nn.init.constant_(sub_m.bias, 0.0)
                        nn.init.constant_(sub_m.weight, 1.0)