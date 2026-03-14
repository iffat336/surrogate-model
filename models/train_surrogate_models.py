"""
Train Surrogate Models: Gaussian Process Regression + Random Forest
for Hygrothermal-Mechanical Coupling Prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

class SurrogateModelTrainer:
    """Train and validate surrogate models for hygrothermal-mechanical coupling"""
    
    def __init__(self, data_path, output_path):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.gp_model = None
        self.rf_model = None
        self.metrics = {}
        
    def load_data(self):
        """Load synthetic FEM dataset"""
        csv_file = Path(__file__).parent.parent / 'data' / 'fem_coupled_hygrothermal_mechanical.csv'
        self.df = pd.read_csv(csv_file)
        print(f"Loaded dataset: {self.df.shape}")
        return self.df
    
    def prepare_data(self):
        """Extract features and targets for modeling"""
        # Input features (multiscale parameters)
        self.feature_cols = [
            'porosity', 'density', 'thermal_cond', 'moisture_diffusivity',
            'E0_dry', 'E_sensitivity_to_moisture', 'nu',
            'RH_exposure', 'temperature', 'thickness',
            'load_magnitude', 'exposure_time_days'
        ]
        
        # Target: Young's modulus at wet conditions
        self.target_col = 'E_effective_wet'
        
        X = self.df[self.feature_cols].values
        y = self.df[self.target_col].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratify split by saturtion level
        saturation_bins = pd.cut(self.df['RH_exposure'], bins=3, labels=['dry', 'medium', 'wet'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42,
            stratify=saturation_bins
        )
        
        print(f"\nTraining set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_gp_model(self):
        """Train Gaussian Process with composite kernel"""
        print("\n" + "="*70)
        print("TRAINING GAUSSIAN PROCESS REGRESSOR")
        print("="*70)
        
        # Composite kernel for hygrothermal-mechanical coupling
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0) + \
                 RBF(length_scale=1.0) + ConstantKernel(0.01)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=15,
            normalize_y=True,
            random_state=42
        )
        
        self.gp_model.fit(self.X_train, self.y_train)
        
        # Predictions with uncertainty
        self.y_pred_gp, self.y_std_gp = self.gp_model.predict(self.X_test, return_std=True)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_gp))
        mae = np.mean(np.abs(self.y_test - self.y_pred_gp))
        r2 = r2_score(self.y_test, self.y_pred_gp)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred_gp)
        
        # Prediction interval coverage
        coverage = np.mean((self.y_test >= self.y_pred_gp - 1.96*self.y_std_gp) &
                          (self.y_test <= self.y_pred_gp + 1.96*self.y_std_gp))
        
        self.metrics['gp'] = {
            'rmse': rmse, 'mae': mae,  'r2': r2, 'mape': mape,
            'coverage': coverage
        }
        
        print(f"GPR Model Performance:")
        print(f"  RMSE: {rmse:.4f} GPa")
        print(f"  MAE:  {mae:.4f} GPa")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape*100:.2f}%")
        print(f"  95% PI Coverage: {coverage*100:.1f}%")
        
        return self.gp_model
    
    def train_rf_model(self):
        """Train Random Forest with feature importance analysis"""
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST REGRESSOR")
        print("="*70)
        
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        self.y_pred_rf = self.rf_model.predict(self.X_test)
        
        # Metrics  
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_rf))
        mae = np.mean(np.abs(self.y_test - self.y_pred_rf))
        r2 = r2_score(self.y_test, self.y_pred_rf)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred_rf)
        
        self.metrics['rf'] = {
            'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape
        }
        
        print(f"Random Forest Performance:")
        print(f"  RMSE: {rmse:.4f} GPa")
        print(f"  MAE:  {mae:.4f} GPa")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape*100:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 6 Most Important Features (Coupled Mechanisms):")
        for idx, row in feature_importance.head(6).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return self.rf_model, feature_importance
    
    def save_models(self):
        """Save trained models to disk"""
        models_dir = self.output_path / 'saved_models'
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / 'gp_model.pkl', 'wb') as f:
            pickle.dump(self.gp_model, f)
        with open(models_dir / 'rf_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        with open(models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✓ Models saved to {models_dir}")
    
    def generate_validation_plots(self, feature_importance):
        """Generate visualization plots"""
        plots_dir = self.output_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Parity plot - GP
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(self.y_test, self.y_pred_gp, alpha=0.6, s=50, label='Predictions')
        axes[0].errorbar(self.y_test, self.y_pred_gp, yerr=1.96*self.y_std_gp,
                        fmt='none', alpha=0.3, ecolor='red', capsize=3, label='95% Confidence')
        min_val, max_val = self.y_test.min(), self.y_test.max()
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('FEM E_effective (GPa)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('GP Predicted E_effective (GPa)', fontsize=11, fontweight='bold')
        axes[0].set_title('Gaussian Process Parity Plot', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Parity plot - RF
        axes[1].scatter(self.y_test, self.y_pred_rf, alpha=0.6, s=50, color='orange')
        axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        axes[1].set_xlabel('FEM E_effective (GPa)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('RF Predicted E_effective (GPa)', fontsize=11, fontweight='bold')
        axes[1].set_title('Random Forest Parity Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / '01_parity_plots.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 01_parity_plots.png")
        plt.close()
        
        # 3. Residuals plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        residuals_gp = self.y_test - self.y_pred_gp
        axes[0].scatter(self.y_pred_gp, residuals_gp, alpha=0.6, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].fill_between(np.sort(self.y_pred_gp), 
                            -1.96*self.y_std_gp[np.argsort(self.y_pred_gp)],
                            1.96*self.y_std_gp[np.argsort(self.y_pred_gp)],
                            alpha=0.2, color='blue', label='95% UQ Band')
        axes[0].set_xlabel('GP Predicted E_effective (GPa)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Residuals (GPa)', fontsize=11, fontweight='bold')
        axes[0].set_title('GP Residuals with Uncertainty Bands', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        residuals_rf = self.y_test - self.y_pred_rf
        axes[1].scatter(self.y_pred_rf, residuals_rf, alpha=0.6, s=50, color='orange')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('RF Predicted E_effective (GPa)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Residuals (GPa)', fontsize=11, fontweight='bold')
        axes[1].set_title('RF Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / '02_residuals_plots.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 02_residuals_plots.png")
        plt.close()
        
        # 4. Feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance.head(10)
        ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Features: Moisture-Mechanical Coupled Mechanisms', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(plots_dir / '03_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 03_feature_importance.png")
        plt.close()
        
        # 5. Model comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        models = ['GP', 'Random Forest']
        r2_scores = [self.metrics['gp']['r2'], self.metrics['rf']['r2']]
        rmse_scores = [self.metrics['gp']['rmse'], self.metrics['rf']['rmse']]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, r2_scores, width, label='R² Score', color='steelblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE (GPa)', color='coral', alpha=0.7)
        
        ax.set_ylabel('R² Score', fontsize=11, fontweight='bold', color='steelblue')
        ax2.set_ylabel('RMSE (GPa)', fontsize=11, fontweight='bold', color='coral')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / '04_model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 04_model_comparison.png")
        plt.close()
        
        print(f"\n✓ All plots saved to {plots_dir}")

if __name__ == "__main__":
    # Initialize trainer
    data_path = Path(__file__).parent
    output_path = data_path.parent / 'results'
    
    trainer = SurrogateModelTrainer(data_path, output_path)
    
    # Training pipeline
    trainer.load_data()
    trainer.prepare_data()
    trainer.train_gp_model()
    rf_model, feature_importance = trainer.train_rf_model()
    trainer.save_models()
    trainer.generate_validation_plots(feature_importance)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE - Models Ready for Deployment")
    print("="*70)
