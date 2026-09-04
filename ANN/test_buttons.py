import sys
import os
import unittest

# Ensure UTF-8 output encoding for Windows terminals
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import app

class TestANNSuiteButtons(unittest.TestCase):
    def setUp(self):
        print("\n--------------------------------------------------")

    def test_01_regression_model_and_metrics(self):
        print("Testing Button 1: Check Housing Regression Saved Model & Metrics...")
        model, scaler_x, scaler_y, meta = app.load_housing_resources()
        self.assertIn('r2', meta)
        self.assertGreater(meta['r2'], 0.90)
        print(f"[PASS] Housing Model Loaded: R2={meta['r2']:.4f} ({meta['r2']*100:.2f}%), MAE=${meta['mae']:,.2f}")

    def test_02_regression_predict_button(self):
        print("Testing Button 2: Predict Housing Price...")
        sample_input = {
            'Square_Feet': 250.0,
            'Num_Bedrooms': 3,
            'Num_Bathrooms': 2,
            'Num_Floors': 2,
            'Year_Built': 1998,
            'Has_Garden': 1,
            'Has_Pool': 0,
            'Garage_Size': 30,
            'Location_Score': 8.5,
            'Distance_to_Center': 4.2
        }
        price = app.predict_regression(sample_input)
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
        print(f"[PASS] Regression Predict Button: Predicted Price = ${price:,.2f}")

    def test_03_binary_model_and_metrics(self):
        print("Testing Button 3: Check IoT Binary Threat Saved Model & Metrics...")
        model, scaler, encoders, meta = app.load_binary_resources()
        self.assertIn('accuracy', meta)
        self.assertGreater(meta['accuracy'], 0.90)
        print(f"[PASS] Binary Model Loaded: Accuracy={meta['accuracy']*100:.2f}%, F1={meta['f1']:.4f}")

    def test_04_binary_predict_button(self):
        print("Testing Button 4: Scan/Predict Cyber Attack Threat...")
        sample_input = {
            'src_ip': '192.168.1.193',
            'src_port': 49180,
            'dst_ip': '192.168.1.37',
            'dst_port': 8080,
            'proto': 'tcp',
            'duration': 0.0084,
            'src_bytes': 101568,
            'dst_bytes': 2592,
            'conn_state': 'SF'
        }
        label, prob = app.predict_binary(sample_input)
        self.assertIn(label, [0, 1])
        self.assertTrue(0.0 <= prob <= 1.0)
        print(f"[PASS] Binary Predict Button: Output Label={label} ({'Attack' if label==1 else 'Normal'}), Probability={prob:.4f}")

    def test_05_multiclass_model_and_metrics(self):
        print("Testing Button 5: Check 10-Class Multiclass Saved Model & Metrics...")
        model, scaler, encoders, target_encoder, meta = app.load_multiclass_resources()
        self.assertIn('accuracy', meta)
        self.assertGreater(meta['accuracy'], 0.85)
        print(f"[PASS] Multiclass Model Loaded: Accuracy={meta['accuracy']*100:.2f}%, F1-Weighted={meta['f1_weighted']:.4f}")

    def test_06_multiclass_predict_button(self):
        print("Testing Button 6: Categorize Network Traffic Type...")
        sample_input = {
            'service': 'http',
            'proto': 'tcp',
            'http_trans_depth': 1,
            'http_request_body_len': 0,
            'http_response_body_len': 0,
            'http_status_code': 200
        }
        top_class, probs = app.predict_multiclass(sample_input)
        self.assertIsInstance(top_class, str)
        self.assertIn(top_class, probs)
        print(f"[PASS] Multiclass Predict Button: Predicted Category='{top_class}', Probabilities={probs}")

if __name__ == '__main__':
    unittest.main()
