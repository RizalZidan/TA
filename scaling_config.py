#!/usr/bin/env python3
"""
APD Violation Scaling Configuration & Demo
Anda bisa mengatur scaling untuk setiap jenis violation
"""

class ScalingConfig:
    def __init__(self):
        """
        Initialize scaling configuration
        """
        # Scaling factors for different violation types
        self.scaling_factors = {
            'No_Helmet': {
                'expand_factor': 0.8,  # Shrink to 80% (kecilin)
                'min_width': 30,      # Minimum width after scaling
                'min_height': 25,     # Minimum height after scaling
                'position_offset': 0.0  # Position offset (0 = center)
            },
            'No_Vest': {
                'expand_factor': 0.9, # Shrink to 90% (kecilin)
                'min_width': 40,      # Minimum width after scaling
                'min_height': 35,     # Minimum height after scaling
                'position_offset': 0.0  # Position offset (0 = center)
            }
        }
        
        # Default configuration
        self.use_smart_scaling = True
        self.show_scaling_info = True
        
        print("📏 Scaling Configuration initialized")
        print(f"📊 No_Helmet: {self.scaling_factors['No_Helmet']}")
        print(f"📊 No_Vest: {self.scaling_factors['No_Vest']}")
    
    def get_scaling_config(self, class_name):
        """
        Get scaling configuration for specific class
        
        Args:
            class_name: 'No_Helmet' or 'No_Vest'
            
        Returns:
            Dictionary with scaling parameters
        """
        return self.scaling_factors.get(class_name, {
            'expand_factor': 1.0,
            'min_width': 50,
            'min_height': 50,
            'position_offset': 0.0
        })
    
    def update_scaling(self, class_name, **kwargs):
        """
        Update scaling parameters for specific class
        
        Args:
            class_name: 'No_Helmet' or 'No_Vest'
            **kwargs: Parameters to update (expand_factor, min_width, min_height, position_offset)
        """
        if class_name in self.scaling_factors:
            self.scaling_factors[class_name].update(kwargs)
            print(f"✅ Updated {class_name} scaling: {self.scaling_factors[class_name]}")
        else:
            print(f"❌ Unknown class: {class_name}")
    
    def apply_custom_scaling(self, bbox, class_name):
        """
        Apply custom scaling based on configuration
        
        Args:
            bbox: [x1, y1, x2, y2] original bounding box
            class_name: 'No_Helmet' or 'No_Vest'
            
        Returns:
            Scaled bounding box
        """
        if not self.use_smart_scaling:
            return bbox
        
        config = self.get_scaling_config(class_name)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Apply expansion factor
        expand_factor = config['expand_factor']
        new_width = max(config['min_width'], int(width * expand_factor))
        new_height = max(config['min_height'], int(height * expand_factor))
        
        # Calculate position with offset
        offset = config['position_offset']
        new_y1 = center_y - new_height // 2 + int(height * offset)
        new_y2 = center_y + new_height // 2 + int(height * offset)
        new_x1 = center_x - new_width // 2
        new_x2 = center_x + new_width // 2
        
        # Ensure bounds are within frame
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def print_current_config(self):
        """Print current scaling configuration"""
        print("\n" + "="*50)
        print("📏 CURRENT SCALING CONFIGURATION")
        print("="*50)
        
        for class_name, config in self.scaling_factors.items():
            print(f"\n🎯 {class_name}:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        
        print(f"\n⚙️ Smart Scaling: {self.use_smart_scaling}")
        print(f"📊 Show Scaling Info: {self.show_scaling_info}")
        print("="*50)
    
    def demo_interactive(self):
        """Interactive demo untuk mengatur scaling"""
        print("\n🎯 INTERACTIVE SCALING DEMO")
        print("=" * 50)
        
        # Tampilkan konfigurasi awal
        print("📋 Konfigurasi Awal:")
        self.print_current_config()
        
        print("\n🔧 Opsi Pengaturan:")
        print("1. Update No_Helmet scaling")
        print("2. Update No_Vest scaling") 
        print("3. Tampilkan konfigurasi")
        print("4. Test scaling")
        print("5. Reset ke default")
        print("6. Keluar")
        
        while True:
            choice = input("\n📝 Pilih opsi (1-6): ").strip()
            
            if choice == '1':
                # Update No_Helmet scaling
                print("\n🔧 Update No_Helmet Scaling:")
                print("   expand_factor (default 0.8): ", end="")
                try:
                    expand = float(input())
                except:
                    expand = 0.8
                
                print("   min_width (default 30): ", end="")
                try:
                    min_w = int(input())
                except:
                    min_w = 30
                    
                print("   min_height (default 25): ", end="")
                try:
                    min_h = int(input())
                except:
                    min_h = 25
                
                print("   position_offset (default 0.0): ", end="")
                try:
                    offset = float(input())
                except:
                    offset = 0.0
                
                self.update_scaling('No_Helmet',
                              expand_factor=expand,
                              min_width=min_w,
                              min_height=min_h,
                              position_offset=offset)
                
            elif choice == '2':
                # Update No_Vest scaling
                print("\n🔧 Update No_Vest Scaling:")
                print("   expand_factor (default 0.9): ", end="")
                try:
                    expand = float(input())
                except:
                    expand = 0.9
                
                print("   min_width (default 40): ", end="")
                try:
                    min_w = int(input())
                except:
                    min_w = 40
                    
                print("   min_height (default 35): ", end="")
                try:
                    min_h = int(input())
                except:
                    min_h = 35
                
                print("   position_offset (default 0.0): ", end="")
                try:
                    offset = float(input())
                except:
                    offset = 0.0
                
                self.update_scaling('No_Vest',
                              expand_factor=expand,
                              min_width=min_w,
                              min_height=min_h,
                              position_offset=offset)
                
            elif choice == '3':
                # Tampilkan konfigurasi
                self.print_current_config()
                
            elif choice == '4':
                # Test scaling
                print("\n🧪 Test Scaling:")
                print("   Input bounding box (x1,y1,x2,y2): ", end="")
                try:
                    bbox_input = input().strip()
                    if bbox_input:
                        parts = bbox_input.split(',')
                        if len(parts) == 4:
                            test_bbox = [int(p.strip()) for p in parts]
                            
                            print(f"\n📏 Test No_Helmet scaling:")
                            scaled_helmet = self.apply_custom_scaling(test_bbox, 'No_Helmet')
                            print(f"   Original: {test_bbox}")
                            print(f"   Scaled: {scaled_helmet}")
                            
                            print(f"\n📏 Test No_Vest scaling:")
                            scaled_vest = self.apply_custom_scaling(test_bbox, 'No_Vest')
                            print(f"   Original: {test_bbox}")
                            print(f"   Scaled: {scaled_vest}")
                        else:
                            print("❌ Format salah! Gunakan: x1,y1,x2,y2")
                except:
                    print("❌ Input tidak valid")
                
            elif choice == '5':
                # Reset ke default
                print("\n🔄 Reset ke konfigurasi default...")
                self.scaling_factors['No_Helmet'] = {
                    'expand_factor': 0.8,
                    'min_width': 30,
                    'min_height': 25,
                    'position_offset': 0.0
                }
                self.scaling_factors['No_Vest'] = {
                    'expand_factor': 0.9,
                    'min_width': 40,
                    'min_height': 35,
                    'position_offset': 0.0
                }
                print("✅ Reset ke konfigurasi default")
            
            elif choice == '6':
                print("\n👋 Keluar demo")
                break
            
            else:
                print("❌ Pilihan tidak valid!")
    
    def show_scaling_examples(self):
        """Tampilkan contoh scaling yang berbeda"""
        print("\n🎯 CONTOH SCALING EXAMPLES:")
        print("=" * 50)
        
        # Test berbagai konfigurasi
        configs = [
            {
                'name': 'Default',
                'no_helmet': {'expand_factor': 0.8, 'min_width': 30, 'min_height': 25},
                'no_vest': {'expand_factor': 0.9, 'min_width': 40, 'min_height': 35}
            },
            {
                'name': 'Conservative',
                'no_helmet': {'expand_factor': 0.6, 'min_width': 25, 'min_height': 20},
                'no_vest': {'expand_factor': 0.7, 'min_width': 35, 'min_height': 30}
            },
            {
                'name': 'Aggressive',
                'no_helmet': {'expand_factor': 1.2, 'min_width': 50, 'min_height': 40},
                'no_vest': {'expand_factor': 1.3, 'min_width': 60, 'min_height': 50}
            },
            {
                'name': 'Minimal',
                'no_helmet': {'expand_factor': 0.5, 'min_width': 20, 'min_height': 15},
                'no_vest': {'expand_factor': 0.6, 'min_width': 25, 'min_height': 20}
            }
        ]
        
        test_bbox = [100, 100, 200, 150]  # 100x50 bbox
        
        for config in configs:
            print(f"\n🎯 {config['name']} Configuration:")
            print(f"   No_Helmet: expand_factor={config['no_helmet']['expand_factor']}, "
                  f"min_width={config['no_helmet']['min_width']}, "
                  f"min_height={config['no_helmet']['min_height']}")
            print(f"   No_Vest: expand_factor={config['no_vest']['expand_factor']}, "
                  f"min_width={config['no_vest']['min_width']}, "
                  f"min_height={config['no_vest']['min_height']}")
            
            # Test No_Helmet
            scaled_helmet = self.apply_custom_scaling(test_bbox, 'No_Helmet')
            print(f"   No_Helmet: {test_bbox} -> {scaled_helmet}")
            
            # Test No_Vest
            scaled_vest = self.apply_custom_scaling(test_bbox, 'No_Vest')
            print(f"   No_Vest: {test_bbox} -> {scaled_vest}")
            
            # Calculate coverage area
            original_area = (test_bbox[2] - test_bbox[0]) * (test_bbox[3] - test_bbox[1])
            helmet_area = (scaled_helmet[2] - scaled_helmet[0]) * (scaled_helmet[3] - scaled_helmet[1])
            vest_area = (scaled_vest[2] - scaled_vest[0]) * (scaled_vest[3] - scaled_vest[1])
            
            print(f"   Coverage Area - Original: {original_area}px²")
            print(f"   Coverage Area - Helmet: {helmet_area}px² ({helmet_area/original_area:.1%})")
            print(f"   Coverage Area - Vest: {vest_area}px² ({vest_area/original_area:.1%})")

# Global configuration instance
scaling_config = ScalingConfig()

if __name__ == "__main__":
    print("🚀 APD Violation Scaling Configuration & Demo Tool")
    print("=" * 50)
    
    print("📋 Pilih mode:")
    print("1. Demo Interaktif")
    print("2. Tampilkan Contoh Scaling")
    print("3. Tampilkan Konfigurasi Saat Ini")
    print("4. Keluar")
    
    choice = input("\n📝 Pilih mode (1-4): ").strip()
    
    if choice == '1':
        scaling_config.demo_interactive()
    elif choice == '2':
        scaling_config.show_scaling_examples()
    elif choice == '3':
        scaling_config.print_current_config()
    elif choice == '4':
        print("👋 Keluar dari demo")
    else:
        print("👋 Keluar")
    
    print("\n🎉 Selesai!")
