import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from backend.websocket import ConnectionManager, manager


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()

        cm = ConnectionManager()
        await cm.connect(mock_ws)

        mock_ws.accept.assert_called_once()
        assert mock_ws in cm.active_connections

    @pytest.mark.asyncio
    async def test_disconnect_removes_websocket(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()

        cm = ConnectionManager()
        await cm.connect(mock_ws)
        cm.disconnect(mock_ws)

        assert mock_ws not in cm.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_connections(self):
        mock_ws1 = MagicMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2 = MagicMock()
        mock_ws2.send_text = AsyncMock()

        cm = ConnectionManager()
        cm.active_connections = [mock_ws1, mock_ws2]

        await cm.broadcast("hello")

        mock_ws1.send_text.assert_called_once_with("hello")
        mock_ws2.send_text.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_broadcast_skips_failed_connections(self):
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock(side_effect=Exception("broken"))

        cm = ConnectionManager()
        cm.active_connections = [mock_ws]

        # Should not raise
        await cm.broadcast("hello")

    @pytest.mark.asyncio
    async def test_broadcast_empty_list(self):
        cm = ConnectionManager()
        cm.active_connections = []

        await cm.broadcast("hello")  # Should not raise


class TestSingleton:
    def test_manager_is_singleton(self):
        from backend.websocket import manager as m1
        from backend.websocket import ConnectionManager as CM

        assert isinstance(m1, CM)
        assert len(m1.active_connections) == 0


class TestWebsocketEndpoint:
    @pytest.mark.asyncio
    async def test_endpoint_disconnects_manager_on_websocket_disconnect(self):
        from backend.api import websocket_endpoint
        from fastapi import WebSocketDisconnect as FSDisconnect

        mock_ws = MagicMock()
        mock_ws.receive_text = AsyncMock(side_effect=FSDisconnect())
        mock_manager = MagicMock()
        mock_manager.connect = AsyncMock()
        mock_manager.disconnect = MagicMock()

        with patch("backend.api.manager", mock_manager):
            try:
                await websocket_endpoint(mock_ws)
            except FSDisconnect:
                pass

        mock_manager.disconnect.assert_called_once_with(mock_ws)

    @pytest.mark.asyncio
    async def test_websocket_disconnect_is_importable(self):
        from fastapi import WebSocketDisconnect
        assert WebSocketDisconnect is not None
