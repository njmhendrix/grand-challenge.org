"use strict";

const timeout = 5000;

const cards = {
    "imageImport": document.getElementById("imageImportCard"),
    "job": document.getElementById("jobCard"),
    "resultImport": document.getElementById("resultImportCard")
};

const averageJobDuration = moment.duration(JSON.parse(document.getElementById("averageJobDuration").textContent));

// Set anything less than 1s to "a few seconds"
moment.relativeTimeThreshold('ss', 1);

function getUploadSessionStatus(statusUrl) {
    // Checks on the status of the Session (queued, running, started, etc)
    fetch(statusUrl)
        .then(response => response.json())
        .then(session => handleUploadSessionStatus(statusUrl, session.status, session.image_set));
}

function handleUploadSessionStatus(statusUrl, status, imageUrls) {
    switch (status.toLowerCase()) {
        case "queued":
        case "re-queued":
            setCardAwaitingMessage(cards.imageImport, status);
            setTimeout(function () {
                getUploadSessionStatus(statusUrl)
            }, Math.floor(Math.random() * timeout) + 100);
            break;
        case "started":
            setCardActiveMessage(cards.imageImport, status);
            setTimeout(function () {
                getUploadSessionStatus(statusUrl)
            }, Math.floor(Math.random() * timeout) + 100);
            break;
        case "succeeded":
            let msg = `Imported ${imageUrls.length} Image`;
            if (imageUrls.length < 1) {
                setCardErrorMessage(cards.imageImport, "No Images Imported");
                return;
            } else if (imageUrls.length > 1) {
                msg += "s"
            }
            setCardCompleteMessage(cards.imageImport, msg);
            getJobsForImages(imageUrls);
            break;
        case "failed":
        case "cancelled":
            setCardErrorMessage(cards.imageImport, status);
            break;
        default:
            setCardErrorMessage(cards.imageImport, "Import error");
    }
}

function getJobsForImages(imageUrls) {
    setCardAwaitingMessage(cards.job, "Fetching Status");

    Promise.all(imageUrls.map(url => fetch(url).then(response => response.json()))
    ).then(images => {
        getJobStatus(images.map(i => i.job_set).flat());
    });
}

function getJobStatus(jobUrls) {
    Promise.all(jobUrls.map(url => fetch(url).then(response => response.json()))
    ).then(jobs => {
        handleJobStatus(jobs);
    });
}

function handleJobStatus(jobs) {
    let jobStatuses = jobs.map(j => j.status.toLowerCase());
    let jobUrls = jobs.map(j => j.api_url);

    let remainingJobs = jobStatuses.filter(s => ["started", "queued"].includes(s)).length;
    let estimatedRemainingTime = moment.duration(remainingJobs * averageJobDuration);

    if (jobStatuses.every(s => s === "succeeded")) {
        setCardCompleteMessage(cards.job, "");
        getResults(jobs.map(j => j.result));
    } else if (jobStatuses.some(s => s === "started")) {
        setCardActiveMessage(cards.job, `Started, ${estimatedRemainingTime.humanize()} remaining`);
        setTimeout(function () {
            getJobStatus(jobUrls)
        }, Math.floor(Math.random() * timeout) + 100);
    } else if (jobStatuses.some(s => s === "queued") || jobStatuses.some(s => s === "re-queued")) {
        setCardAwaitingMessage(cards.job, "Queued");
        setTimeout(function () {
            getJobStatus(jobUrls)
        }, Math.floor(Math.random() * timeout) + 100);
    } else {
        setCardErrorMessage(cards.job, "Errored");
    }
}

function getResults(resultUrls) {
    setCardAwaitingMessage(cards.resultImport, "Fetching Status");

    Promise.all(resultUrls.map(url => fetch(url).then(response => response.json()))
    ).then(results => {
        let resultImportSessionUrls = results.map(r => r.import_session).filter(s => s !== null);
        if (resultImportSessionUrls.length > 0) {
            getResultImportStatus(resultImportSessionUrls);
        } else {
            setCardCompleteMessage(cards.resultImport, "");
        }
    });
}

function getResultImportStatus(resultImportSessionUrls) {
    Promise.all(resultImportSessionUrls.map(url => fetch(url).then(response => response.json()))
    ).then(importSessions => {
        handleImportSessionsStatus(importSessions);
    });
}

function handleImportSessionsStatus(resultImportSessions) {
    let importSessionStatuses = resultImportSessions.map(s => s.status.toLowerCase());
    let importSessionUrls = resultImportSessions.map(s => s.api_url);

    if (importSessionStatuses.every(s => s === "succeeded")) {
        setCardCompleteMessage(cards.resultImport, "View Results");
    } else if (importSessionStatuses.some(s => s === "started")) {
        setCardActiveMessage(cards.resultImport, "Started");
        setTimeout(function () {
            getResultImportStatus(importSessionUrls)
        }, Math.floor(Math.random() * timeout) + 100);
    } else if (importSessionStatuses.some(s => s === "queued") || importSessionStatuses.some(s => s === "re-queued")) {
        setCardAwaitingMessage(cards.resultImport, "Queued");
        setTimeout(function () {
            getResultImportStatus(importSessionUrls)
        }, Math.floor(Math.random() * timeout) + 100);
    } else {
        setCardErrorMessage(cards.resultImport, "Errored");
    }
}

function setCardAwaitingMessage(card, msg) {
    card.classList.replace("border-light", "border-primary");
    card.classList.remove("text-muted", "active");
    card.querySelector(".statusMessage").innerHTML = msg;
    card.querySelector(".statusSymbol").innerHTML = "<div class=\"text-secondary spinner-grow\" role=\"status\"><span class=\"sr-only\">Loading...</span></div>";
}

function setCardActiveMessage(card, msg) {
    if (!card.classList.contains("active")) {
        card.classList.add("active");
        card.querySelector(".statusSymbol").innerHTML = "<div class=\"text-primary spinner-border\" role=\"status\"><span class=\"sr-only\">Loading...</span></div>";
    }
    card.querySelector(".statusMessage").innerHTML = msg;
}

function setCardCompleteMessage(card, msg) {
    card.classList.remove("active");
    card.classList.replace("border-primary", "border-success");
    card.querySelector(".statusMessage").innerHTML = msg;
    card.querySelector(".statusSymbol").innerHTML = "<i class=\"text-success fa fa-check fa-2x\"></i>";
}

function setCardErrorMessage(card, msg) {
    card.classList.remove("active", "text-muted", "border-primary", "border-success");
    card.classList.add("border-danger");
    card.querySelector(".statusMessage").innerHTML = msg;
    card.querySelector(".statusSymbol").innerHTML = "<i class=\"text-danger fa fa-times fa-2x\"></i>";
}