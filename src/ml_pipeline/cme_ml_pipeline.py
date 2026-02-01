#!/usr/bin/env python3
"""
CME Arrival Prediction Machine Learning Pipeline
Supports training on ISRO dataset and testing on ACE/WIND data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CMEArrivalPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = {}
        
    def load_data(self, filepath='balanced_cme_prediction_dataset_final.csv'):
        """Load and prepare the CME dataset"""
        print("=== LOADING CME DATASET ===")
        
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        
        # Check target distribution
        if 'has_cme' in self.df.columns:
            target_dist = self.df['has_cme'].value_counts().sort_index()
            print(f"Target distribution:")
            for value, count in target_dist.items():
                percentage = (count / len(self.df)) * 100
                print(f"  has_cme={value}: {count:,} ({percentage:.1f}%)")
        
        return self.df
    
    def prepare_features(self, target_col='has_cme'):
        """Prepare features for machine learning"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Define feature categories
        self.feature_categories = {
            'time_features': ['cme_time_unix'],
            'solar_wind_features': [
                'proton_density_mean', 'alpha_density_mean', 'proton_temperature_mean',
                'proton_bulk_speed_mean', 'alpha_proton_ratio_mean'
            ],
            'cme_properties': ['cme_velocity', 'cme_width', 'pa'],
            'cme_labels': ['halo_class', 'earth_directed', 'theoretical_transit_hours']
        }
        
        # Collect available features
        available_features = []
        for category, features in self.feature_categories.items():
            available = [f for f in features if f in self.df.columns]
            available_features.extend(available)
            print(f"{category}: {len(available)}/{len(features)} available")
        
        print(f"\nTotal features available: {len(available_features)}")
        
        # Create feature matrix
        X = self.df[available_features].copy()
        y = self.df[target_col].copy()
        
        # Handle missing values
        print(f"\nHandling missing values...")
        missing_before = X.isnull().sum().sum()
        
        # For solar wind features, use forward fill + median
        for feature in self.feature_categories['solar_wind_features']:
            if feature in X.columns:
                X[feature] = X[feature].fillna(method='ffill').fillna(X[feature].median())
        
        # For CME properties, use -999 for missing (indicates no CME)
        for feature in self.feature_categories['cme_properties']:
            if feature in X.columns:
                X[feature] = X[feature].fillna(-999)
        
        # For other features, use appropriate defaults
        for feature in self.feature_categories['cme_labels']:
            if feature in X.columns:
                if feature == 'halo_class':
                    X[feature] = X[feature].fillna(0)
                elif feature == 'earth_directed':
                    X[feature] = X[feature].fillna(0)
                else:
                    X[feature] = X[feature].fillna(X[feature].median())
        
        missing_after = X.isnull().sum().sum()
        print(f"Missing values: {missing_before} -> {missing_after}")
        
        self.feature_names = list(X.columns)
        self.X = X
        self.y = y
        
        print(f"Final feature matrix: {X.shape}")
        print(f"Features: {self.feature_names}")
        
        return X, y
    
    def split_data(self, test_size=0.2, stratify=True):
        """Split data into train/test sets"""
        print(f"\n=== DATA SPLITTING ===")
        
        stratify_target = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_target
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Check class distribution in splits
        train_dist = self.y_train.value_counts().sort_index()
        test_dist = self.y_test.value_counts().sort_index()
        
        print(f"Training distribution: {train_dist.to_dict()}")
        print(f"Test distribution: {test_dist.to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print(f"\n=== FEATURE SCALING ===")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Features scaled using StandardScaler")
        print(f"Training mean: {self.X_train_scaled.mean():.4f}")
        print(f"Training std: {self.X_train_scaled.std():.4f}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def initialize_models(self):
        """Initialize multiple ML models"""
        print(f"\n=== INITIALIZING MODELS ===")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"Class weights: {class_weight_dict}")
        
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            ),
            'SVM': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_models(self):
        """Train all models"""
        print(f"\n=== TRAINING MODELS ===")
        
        self.trained_models = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for models that benefit from scaling
            if name in ['LogisticRegression', 'SVM']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'auc_score': auc_score,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  AUC Score: {auc_score:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.results
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print(f"\n=== MODEL EVALUATION ===")
        
        # Create comparison dataframe
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'AUC_Score': result['auc_score'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC_Score', ascending=False)
        
        print("Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model_result = self.results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"AUC Score: {best_model_result['auc_score']:.4f}")
        
        # Detailed evaluation for best model
        print(f"\nDetailed Classification Report ({best_model_name}):")
        print(classification_report(self.y_test, best_model_result['y_pred']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_model_result['y_pred'])
        print(f"\nConfusion Matrix ({best_model_name}):")
        print(cm)
        
        return comparison_df, best_model_name
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        print(f"\n=== CREATING PERFORMANCE PLOTS ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. AUC Score Comparison
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        
        axes[0, 0].bar(model_names, auc_scores)
        axes[0, 0].set_title('Model AUC Score Comparison')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. ROC Curves
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {result['auc_score']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        
        # 3. Cross-validation scores
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        axes[1, 0].set_title('Cross-Validation Scores')
        axes[1, 0].set_ylabel('CV Score (AUC)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Feature importance (for best model - if available)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_model = self.results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            axes[1, 1].bar(range(len(indices)), importances[indices])
            axes[1, 1].set_title(f'Top 10 Feature Importances ({best_model_name})')
            axes[1, 1].set_ylabel('Importance')
            axes[1, 1].set_xticks(range(len(indices)))
            axes[1, 1].set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
        else:
            axes[1, 1].text(0.5, 0.5, f'{best_model_name}\nno feature importance', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance Not Available')
        
        plt.tight_layout()
        plt.savefig('cme_model_performance.png', dpi=300, bbox_inches='tight')
        print("Performance plots saved as 'cme_model_performance.png'")
        
        return fig
    
    def save_best_model(self, filepath='best_cme_model.pkl'):
        """Save the best performing model"""
        import pickle
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_model = self.results[best_model_name]['model']
        
        model_package = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': best_model_name,
            'performance': self.results[best_model_name]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n✅ Best model ({best_model_name}) saved as '{filepath}'")
        return filepath
    
    def predict_ace_wind_data(self, ace_wind_filepath):
        """Make predictions on ACE/WIND data"""
        print(f"\n=== TESTING ON ACE/WIND DATA ===")
        
        try:
            # Load ACE/WIND data
            ace_data = pd.read_csv(ace_wind_filepath)
            print(f"ACE/WIND data shape: {ace_data.shape}")
            
            # Prepare features (map to same feature names)
            ace_features = self.prepare_ace_wind_features(ace_data)
            
            # Get best model
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
            best_model = self.results[best_model_name]['model']
            
            # Scale features if needed
            if best_model_name in ['LogisticRegression', 'SVM']:
                ace_features_scaled = self.scaler.transform(ace_features)
                predictions = best_model.predict_proba(ace_features_scaled)[:, 1]
            else:
                predictions = best_model.predict_proba(ace_features)[:, 1]
            
            # Add predictions to dataframe
            ace_data['cme_arrival_probability'] = predictions
            ace_data['cme_prediction'] = (predictions > 0.5).astype(int)
            
            print(f"Predictions completed!")
            print(f"Mean CME probability: {predictions.mean():.4f}")
            print(f"Predicted CMEs: {(predictions > 0.5).sum()}/{len(predictions)}")
            
            # Save results
            output_file = 'ace_wind_cme_predictions.csv'
            ace_data.to_csv(output_file, index=False)
            print(f"Results saved as '{output_file}'")
            
            return ace_data, predictions
            
        except Exception as e:
            print(f"Error processing ACE/WIND data: {e}")
            return None, None
    
    def prepare_ace_wind_features(self, ace_data):
        """Prepare ACE/WIND data features to match training data format"""
        print("Mapping ACE/WIND features to training format...")
        
        # Feature mapping dictionary (adjust based on ACE/WIND column names)
        feature_mapping = {
            # Time features
            'time_unix': 'cme_time_unix',
            'epoch': 'cme_time_unix',
            'timestamp': 'cme_time_unix',
            
            # Solar wind features
            'proton_density': 'proton_density_mean',
            'np': 'proton_density_mean',
            'n_p': 'proton_density_mean',
            
            'alpha_density': 'alpha_density_mean',
            'na': 'alpha_density_mean',
            'n_a': 'alpha_density_mean',
            
            'proton_temperature': 'proton_temperature_mean',
            'tp': 'proton_temperature_mean',
            't_p': 'proton_temperature_mean',
            
            'bulk_speed': 'proton_bulk_speed_mean',
            'speed': 'proton_bulk_speed_mean',
            'v': 'proton_bulk_speed_mean',
            'vp': 'proton_bulk_speed_mean',
            
            'alpha_proton_ratio': 'alpha_proton_ratio_mean',
            'na_np': 'alpha_proton_ratio_mean',
        }
        
        # Create feature dataframe
        ace_features = pd.DataFrame()
        
        for ace_col in ace_data.columns:
            ace_col_lower = ace_col.lower().strip()
            if ace_col_lower in feature_mapping:
                target_feature = feature_mapping[ace_col_lower]
                if target_feature in self.feature_names:
                    ace_features[target_feature] = ace_data[ace_col]
        
        # Fill missing features with defaults
        for feature in self.feature_names:
            if feature not in ace_features.columns:
                if 'cme_' in feature:
                    ace_features[feature] = -999  # No CME properties in real-time data
                else:
                    ace_features[feature] = 0  # Default for other features
        
        # Ensure correct column order
        ace_features = ace_features[self.feature_names]
        
        print(f"ACE/WIND features prepared: {ace_features.shape}")
        return ace_features

def main():
    """Main execution function"""
    print("🚀 CME ARRIVAL PREDICTION MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CMEArrivalPredictor()
    
    # Load and prepare data
    predictor.load_data()
    X, y = predictor.prepare_features()
    predictor.split_data()
    predictor.scale_features()
    
    # Train models
    predictor.initialize_models()
    predictor.train_models()
    
    # Evaluate and compare
    comparison_df, best_model = predictor.evaluate_models()
    predictor.plot_model_comparison()
    
    # Save best model
    model_filepath = predictor.save_best_model()
    
    print(f"\n🎯 TRAINING COMPLETED!")
    print(f"Best model: {best_model}")
    print(f"Model saved: {model_filepath}")
    print(f"Performance plots: cme_model_performance.png")
    
    # Test on ACE/WIND data (if available)
    print(f"\n📡 READY FOR ACE/WIND TESTING")
    print("To test on ACE/WIND data, call:")
    print("predictor.predict_ace_wind_data('your_ace_wind_data.csv')")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
