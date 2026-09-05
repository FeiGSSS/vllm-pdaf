// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! PAP-only Python bindings to upstream Dynamo's local selector.
//! Routing/indexing remain upstream; reservations have explicit owner lifetime.

use dynamo_kv_router::config::try_kv_router_config_from_dynamo_env;
use dynamo_kv_router::services::selection::{
    OverlapScoresRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore,
    SelectionError, WorkerRequest,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

fn error(err: SelectionError) -> PyErr {
    match err {
        SelectionError::BadRequest(message) => PyValueError::new_err(message),
        SelectionError::NotFound(message) => PyKeyError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn output<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<PyObject> {
    pythonize(py, value).map(|v| v.unbind()).map_err(Into::into)
}

#[pyclass]
struct SelectionService {
    inner: Option<Arc<SelectionCore>>,
}

impl SelectionService {
    fn core(&self) -> PyResult<Arc<SelectionCore>> {
        self.inner
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("selector is shut down"))
    }
}

#[pymethods]
impl SelectionService {
    #[new]
    #[pyo3(signature = (*, indexer_threads = 4))]
    fn new(py: Python<'_>, indexer_threads: usize) -> PyResult<Self> {
        if indexer_threads == 0 {
            return Err(PyValueError::new_err("indexer_threads must be positive"));
        }
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
        let config = try_kv_router_config_from_dynamo_env().map_err(PyValueError::new_err)?;
        let core = py
            .allow_threads(|| {
                let _runtime = pyo3_async_runtimes::tokio::get_runtime().enter();
                SelectionCore::try_new_local(
                    config,
                    indexer_threads,
                    CancellationToken::new(),
                    SelectionCacheConfig::default(),
                )
            })
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self {
            inner: Some(Arc::new(core.with_explicit_lifecycle())),
        })
    }

    #[getter]
    fn reservation_lifetime(&self) -> &'static str {
        "explicit-owner-v1"
    }

    fn upsert_worker<'p>(&self, py: Python<'p>, worker: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let req: WorkerRequest = depythonize(worker.bind(py))?;
        let core = self.core()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let record = core.upsert_worker(req).await.map_err(error)?;
            Python::with_gil(|py| output(py, &record))
        })
    }

    fn select_and_reserve<'p>(
        &self,
        py: Python<'p>,
        request: PyObject,
    ) -> PyResult<Bound<'p, PyAny>> {
        let req: SelectAndReserveRequest = depythonize(request.bind(py))?;
        let core = self.core()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = core.select_and_reserve(req).await.map_err(error)?;
            Python::with_gil(|py| output(py, &result))
        })
    }

    fn overlap_scores<'p>(&self, py: Python<'p>, request: PyObject) -> PyResult<Bound<'p, PyAny>> {
        let req: OverlapScoresRequest = depythonize(request.bind(py))?;
        let core = self.core()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = core.overlap_scores(req).await.map_err(error)?;
            Python::with_gil(|py| output(py, &result))
        })
    }

    fn prefill_complete<'p>(
        &self,
        py: Python<'p>,
        request_id: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let core = self.core()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            core.prefill_complete(&request_id).await.map_err(error)
        })
    }

    fn free_reservation<'p>(
        &self,
        py: Python<'p>,
        request_id: String,
    ) -> PyResult<Bound<'p, PyAny>> {
        let core = self.core()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            core.free_reservation(&request_id).await.map_err(error)
        })
    }

    fn ready(&self, py: Python<'_>) -> PyResult<PyObject> {
        output(py, &self.core()?.ready())
    }

    #[pyo3(signature = (*, model_name = None))]
    fn loads(&self, py: Python<'_>, model_name: Option<String>) -> PyResult<PyObject> {
        output(py, &self.core()?.loads(model_name.as_deref(), None))
    }

    fn stop_scheduling(&self) {
        if let Some(core) = &self.inner {
            core.shutdown();
        }
    }

    fn shutdown(&mut self, py: Python<'_>) {
        if let Some(core) = self.inner.take() {
            py.allow_threads(|| {
                core.shutdown();
                drop(core);
            });
        }
    }
}

impl Drop for SelectionService {
    fn drop(&mut self) {
        if let Some(core) = self.inner.take() {
            core.shutdown();
        }
    }
}

#[pymodule]
fn pap_dynamo_router(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SelectionService>()?;
    module.add(
        "UPSTREAM_REVISION",
        "2112d6ba74da72e2715ae69f4b76458b7691380d",
    )?;
    Ok(())
}
