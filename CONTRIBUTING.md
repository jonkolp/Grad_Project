# Contributing to the CXR Lung Disease Research Framework

Thank you for your interest in contributing to this research initiative! We welcome contributions from researchers, clinical professionals, and software engineers.

---

## 🔬 Types of Contributions

We are especially interested in:
- **Model Improvements**: Integrating modern Vision Transformers (Swin, ConvNeXt) or medical foundation models (RadImageNet, BioViL).
- **Explainability**: Adding Grad-CAM, Grad-CAM++, or HiResCAM interpretability backends.
- **Evaluation**: Adding standard evaluation metrics (AUROC, macro/micro F1, PR curves) and clinical dataset benchmarks.
- **DICOM Integration**: Enhancing direct 16-bit DICOM file loaders and PACS interoperability.
- **Bug Fixes & Refactoring**: Clean, well-tested code improvements.

---

## 🛠️ Development Workflow

1. **Fork & Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Set Up Development Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Coding Standards**:
   - Follow **PEP 8** style guidelines.
   - Include clear type annotations (`typing`) and docstrings for all new functions and classes.
   - Ensure inference functions are deterministic (no stochastic noise during inference).
4. **Testing**:
   Verify that scripts compile and run without errors:
   ```bash
   python -m py_compile src/*.py GUI/*.py
   ```
5. **Submitting a Pull Request**:
   - Provide a clear PR description detailing your methodology, experimental rationale, and any performance differences.
