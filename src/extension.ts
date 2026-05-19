import * as path from 'path';
import * as fs from 'fs';
import { workspace, ExtensionContext, window } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

function findPython(configured: string): string | undefined {
    if (configured) {
        return fs.existsSync(configured) ? configured : undefined;
    }
    const candidates = process.platform === 'win32'
        ? ['python', 'python3']
        : ['python3', 'python'];
    for (const cmd of candidates) {
        try {
            require('child_process').execSync(`${cmd} --version`, { stdio: 'ignore' });
            return cmd;
        } catch {
            // try next
        }
    }
    return undefined;
}

export function activate(context: ExtensionContext) {
    const config = workspace.getConfiguration('swaglang');
    const configuredPython = config.get<string>('pythonPath', '');
    const python = findPython(configuredPython);

    if (!python) {
        window.showErrorMessage(
            'SwagLang: Python not found. Install Python 3 or set swaglang.pythonPath in settings.'
        );
        return;
    }

    const serverScript = context.asAbsolutePath(path.join('server', 'server.py'));

    const serverOptions: ServerOptions = {
        command: python,
        args: [serverScript],
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'swaglang' }],
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.swag'),
        },
    };

    client = new LanguageClient(
        'swaglang',
        'SwagLang Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}
